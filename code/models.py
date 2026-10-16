"""
Model definitions (library file - imported by train_cv.py, never run directly).

  SignalEncoder         : one encoder per biosignal, tuned to that signal
                            EDA : 8x average-pool (256 -> 32 Hz) + slope channel -> Conv -> BiLSTM -> attention
                            ECG : strided Conv stem -> BiLSTM -> attention
                            EMG : raw + |raw| (envelope) channels -> strided Conv stem -> BiLSTM -> attention
  VideoEncoder          : CACHED frozen ConvNeXt V2 frame embeddings (K, D) -> BiLSTM -> attention
  FineTuneVideoEncoder  : face frames -> ConvNeXt V2 whose LAST stage(s) are trained -> BiLSTM -> attention
  GatedFusion           : learned per-feature gate over the branch vectors (Thiam et al. 2021)
  MultiModalNet         : branches + gate + classifier, with an auxiliary classifier per branch
  multitask_loss        : fused + per-branch cross-entropy, optional ordinal smoothing / regression term
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGNAL_INDEX = {"eda": 0, "ecg": 1, "emg": 2}


class AttnPool(nn.Module):
    """alpha_t = softmax_t( v^T tanh(W h_t) ),   out = sum_t alpha_t h_t"""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.w = nn.Linear(dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, h):                          # h: (B, L, dim)
        a = torch.softmax(self.v(torch.tanh(self.w(h))).squeeze(-1), dim=1)
        return (a.unsqueeze(-1) * h).sum(1)


class SignalEncoder(nn.Module):
    def __init__(self, kind="ecg", out_dim=128, conv=32, hidden=64, layers=1, dropout=0.25):
        super().__init__()
        self.kind = kind
        if kind == "eda":                          # slow signal: pool to 32 Hz, add slope, no extra striding
            in_ch, k, strides = 2, (7, 5), (1, 1)
        elif kind == "emg":                        # raw + rectified (envelope) channels
            in_ch, k, strides = 2, (11, 7), (4, 2)
        else:                                      # ecg
            in_ch, k, strides = 1, (11, 7), (4, 2)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, conv, k[0], stride=strides[0], padding=k[0] // 2), nn.BatchNorm1d(conv), nn.ELU(),
            nn.Conv1d(conv, conv, k[1], stride=strides[1], padding=k[1] // 2), nn.BatchNorm1d(conv), nn.ELU(),
        )
        self.lstm = nn.LSTM(conv, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.pool = AttnPool(2 * hidden)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, out_dim), nn.ELU(), nn.Dropout(dropout))

    def forward(self, x):                          # x: (B, 1, L)
        if self.kind == "eda":
            x = F.avg_pool1d(x, 8)
            slope = torch.cat([x[:, :, 1:] - x[:, :, :-1], torch.zeros_like(x[:, :, :1])], dim=2)
            x = torch.cat([x, slope], dim=1)
        elif self.kind == "emg":
            x = torch.cat([x, x.abs()], dim=1)
        h = self.stem(x).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.proj(self.pool(h))


class VideoEncoder(nn.Module):
    """Temporal head on top of per-frame features (K, D). Used for cached embeddings and after fine-tuning."""

    def __init__(self, in_dim=768, out_dim=128, proj=256, hidden=128, dropout=0.25):
        super().__init__()
        self.pre = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, proj), nn.ELU())
        self.lstm = nn.LSTM(proj, hidden, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2 * hidden)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, out_dim), nn.ELU(), nn.Dropout(dropout))

    def forward(self, x):                          # x: (B, K, D)
        h, _ = self.lstm(self.pre(x))
        return self.proj(self.pool(h))


class FineTuneVideoEncoder(nn.Module):
    """ConvNeXt V2 on face frames. Everything is frozen except the last `unfreeze` stages and the head norm.
    Frozen part runs under no_grad (saves memory/time); backbone runs in fp16 autocast on the GPU."""

    def __init__(self, name="convnextv2_tiny.fcmae_ft_in22k_in1k", unfreeze=1, pretrained=True,
                 out_dim=128, dropout=0.25, drop_path=0.1):
        super().__init__()
        import timm
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))
        n = len(self.backbone.stages)
        self.unfreeze = max(1, min(int(unfreeze), n))
        for p in self.backbone.parameters():
            p.requires_grad = False
        for s in self.backbone.stages[n - self.unfreeze:]:
            for p in s.parameters():
                p.requires_grad = True
        for p in self.backbone.head.parameters():
            p.requires_grad = True
        self.temporal = VideoEncoder(self.backbone.num_features, out_dim, dropout=dropout)

    def train(self, mode=True):
        super().train(mode)
        n = len(self.backbone.stages)
        self.backbone.stem.eval()                                  # frozen parts never change behaviour
        for s in self.backbone.stages[:n - self.unfreeze]:
            s.eval()
        return self

    def forward(self, x):                          # x: (B, K, 3, S, S), values 0..255 (uint8 or float)
        B, K = x.shape[:2]
        x = x.flatten(0, 1).float() / 255.0
        x = (x - self.mean) / self.std
        bb, n = self.backbone, len(self.backbone.stages)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda):
            with torch.no_grad():
                h = bb.stem(x)
                for s in bb.stages[:n - self.unfreeze]:
                    h = s(h)
            for s in bb.stages[n - self.unfreeze:]:
                h = s(h)
            f = bb.forward_head(bb.norm_pre(h), pre_logits=True)
        return self.temporal(f.float().view(B, K, -1))


class GatedFusion(nn.Module):
    """h_hat_i = tanh(W_i h_i + b_i);  w = softmax(W_g [h_hat_1; ...; h_hat_n]);  h = sum_i w_i * h_i
    The softmax runs over all n*eta features (as in Thiam et al. 2021). A LayerNorm on the output
    restores the scale, because each weight is only ~1/(n*eta)."""

    def __init__(self, n, eta):
        super().__init__()
        self.n, self.eta = n, eta
        self.norm = nn.ModuleList([nn.Linear(eta, eta) for _ in range(n)])
        self.gate = nn.Linear(n * eta, n * eta)
        self.out_norm = nn.LayerNorm(eta)

    def forward(self, hs):
        hn = torch.cat([torch.tanh(l(h)) for l, h in zip(self.norm, hs)], dim=-1)
        w = torch.softmax(self.gate(hn), dim=-1).view(-1, self.n, self.eta)
        fused = (w * torch.stack(hs, dim=1)).sum(1)
        return self.out_norm(fused), w


class MultiModalNet(nn.Module):
    """modalities: any of eda, ecg, emg, video (cached embeddings), videoft (fine-tuned ConvNeXt on frames)."""

    def __init__(self, modalities, video_dim=768, n_classes=5, eta=128, dropout=0.25,
                 ft_model="convnextv2_tiny.fcmae_ft_in22k_in1k", ft_unfreeze=1, ft_pretrained=True):
        super().__init__()
        self.modalities = list(modalities)
        self.enc = nn.ModuleDict()
        for m in self.modalities:
            if m == "video":
                self.enc[m] = VideoEncoder(video_dim, eta, dropout=dropout)
            elif m == "videoft":
                self.enc[m] = FineTuneVideoEncoder(ft_model, ft_unfreeze, ft_pretrained, eta, dropout)
            else:
                self.enc[m] = SignalEncoder(m, eta, dropout=dropout)
        self.aux = nn.ModuleDict({m: nn.Linear(eta, n_classes) for m in self.modalities})
        n = len(self.modalities)
        self.fusion = GatedFusion(n, eta) if n > 1 else None
        self.head = nn.Sequential(nn.Linear(eta, 128), nn.ELU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, sig=None, vid=None):
        hs = []
        for m in self.modalities:
            if m in ("video", "videoft"):
                hs.append(self.enc[m](vid))
            else:
                hs.append(self.enc[m](sig[:, SIGNAL_INDEX[m]:SIGNAL_INDEX[m] + 1]))
        gate = None
        h = hs[0]
        if self.fusion is not None:
            h, gate = self.fusion(hs)
        out = {"fused": self.head(h)}
        if len(self.modalities) > 1:
            for m, hm in zip(self.modalities, hs):
                out[m] = self.aux[m](hm)
        return out, gate


def _soft_ordinal_targets(y, n_classes, sigma, smoothing):
    """Gaussian ordinal label smoothing: target_j ~ exp(-(j - y)^2 / (2 sigma^2)), so neighbours of the true
    level get some probability and far-away levels get almost none."""
    j = torch.arange(n_classes, device=y.device).float().unsqueeze(0)
    t = torch.exp(-((j - y.float().unsqueeze(1)) ** 2) / (2 * sigma ** 2))
    t = t / t.sum(1, keepdim=True)
    if smoothing > 0:
        t = (1 - smoothing) * t + smoothing / n_classes
    return t


def multitask_loss(out, y, n_classes=5, sigma=0.0, reg_weight=0.0, label_smoothing=0.0, fused_weight=0.4):
    """L = fused_weight * L_fused + (1 - fused_weight)/n_branches * sum_i L_i   (Thiam 2021: 0.4 + 4 x 0.15)
       each L = cross-entropy (plain, label-smoothed, or Gaussian-ordinal when sigma > 0)
       + reg_weight * SmoothL1(expected class index, true index) on the fused head (ordinal regression term)."""

    def ce(z):
        if sigma > 0 and n_classes > 2:
            t = _soft_ordinal_targets(y, n_classes, sigma, label_smoothing)
            return -(t * F.log_softmax(z, dim=1)).sum(1).mean()
        return F.cross_entropy(z, y, label_smoothing=label_smoothing)

    fused = ce(out["fused"])
    if reg_weight > 0 and n_classes > 2:
        p = torch.softmax(out["fused"], dim=1)
        expect = (p * torch.arange(n_classes, device=y.device).float()).sum(1)
        fused = fused + reg_weight * F.smooth_l1_loss(expect, y.float())
    branches = [k for k in out if k != "fused"]
    if not branches:
        return fused
    lam = (1.0 - fused_weight) / len(branches)
    return fused_weight * fused + lam * sum(ce(out[k]) for k in branches)