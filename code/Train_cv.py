"""
Stage 2 - subject-grouped k-fold cross-validation on BioVid Part A.

Two tasks (same subject folds, so they are directly comparable):
  5class : BL1 / PA1 / PA2 / PA3 / PA4
  binary : BL1 vs PA4

Modality runs (--modalities, comma separated, '+' joins channels inside one run):
  eda  ecg  emg            single biosignal
  bio                      eda+ecg+emg                      (default)
  video                    frozen ConvNeXt V2, cached embeddings
  videoft                  ConvNeXt V2 with its last stage(s) FINE-TUNED on face frames  (needs --steps frames)
  both                     bio + video      bioft = bio + videoft
  e.g.  --modalities bio,bio+video,bio+videoft,eda,emg,ecg

Per fold:  train subjects -> train; `--val_subjects` other subjects -> early stopping and threshold tuning;
test subjects -> only evaluated.  Reported twice: raw (argmax) and cal (class offsets tuned on the
validation subjects only, never on test).

Examples
  python train_cv.py                                              # bio, both tasks, 15 folds
  python train_cv.py --modalities bio,eda,emg,ecg --tasks binary  # per-channel study
  python train_cv.py --modalities videoft --epochs 15 --batch 32 --ft_lr 5e-5
  python train_cv.py --modalities bio --ensemble 3                # 3 seeds averaged per fold
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from models import MultiModalNet, multitask_loss

TASKS = {
    "5class": dict(classes=["BL1", "PA1", "PA2", "PA3", "PA4"], map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4}),
    "binary": dict(classes=["BL1", "PA4"], map={0: 0, 4: 1}),
}
BIO = ["eda", "ecg", "emg"]
VALID = {"eda", "ecg", "emg", "video", "videoft"}
GROUPS = {"bio": BIO, "biosignals": BIO, "both": BIO + ["video"], "all": BIO + ["video"], "bioft": BIO + ["videoft"]}


def parse_runs(arg):
    runs = []
    for spec in [t.strip().lower() for t in arg.split(",") if t.strip()]:
        mods = []
        for tok in spec.split("+"):
            for c in GROUPS.get(tok.strip(), [tok.strip()]):
                if c not in VALID:
                    raise SystemExit(f"unknown modality '{c}' in '{spec}'. Use {sorted(VALID | set(GROUPS))}")
                if c not in mods:
                    mods.append(c)
        if "video" in mods and "videoft" in mods:
            raise SystemExit(f"'{spec}': choose either video or videoft, not both")
        runs.append((spec, mods))
    return runs


def make_folds(subjects, n_folds, seed):
    subs = np.array(sorted(subjects))
    np.random.RandomState(seed).shuffle(subs)
    return [list(a) for a in np.array_split(subs, n_folds)]


# --------------------------------------------------------------------------- #
class Data:
    def __init__(self, cache, needed, device):
        cache = Path(cache)
        self.meta = pd.read_csv(cache / "meta.csv")
        self.y_full = self.meta.label.values
        self.device = device
        self.sig = self.vid = self.frames = None
        if any(m in BIO for m in needed):
            self.sig = torch.from_numpy(np.load(cache / "signals.npy")).to(device)
        if "video" in needed:
            done = np.load(cache / "emb_done.npy")
            if not done.all():
                raise SystemExit(f"embeddings incomplete ({int(done.sum())}/{len(done)}): re-run prepare_data.py")
            self.vid = torch.from_numpy(np.load(cache / "embeddings.npy")).to(device)   # float16
        if "videoft" in needed:
            if not (cache / "frames.npy").exists():
                raise SystemExit("frames.npy not found: run  python prepare_data.py --steps frames")
            done = np.load(cache / "frames_done.npy")
            if not done.all():
                raise SystemExit(f"frames incomplete ({int(done.sum())}/{len(done)}): re-run prepare_data.py --steps frames")
            self.frames = np.load(cache / "frames.npy", mmap_mode="r")                  # (N, K, 3, S, S) uint8
        self.y = self.mask = None

    def set_task(self, task):
        lut = np.full(5, -1, dtype=np.int64)
        for src, dst in TASKS[task]["map"].items():
            lut[src] = dst
        y = lut[self.y_full]
        self.mask = y >= 0
        self.y = torch.tensor(y, dtype=torch.long, device=self.device)

    def batch(self, idx, train, args, vmode):
        idx = np.asarray(idx)
        idx_t = torch.as_tensor(idx, device=self.device)
        sig = vid = None
        if self.sig is not None:
            x = self.sig[idx_t]                                    # (B, 3, T)
            B, C, T = x.shape
            L = min(args.crop, T)
            if train:                                              # random window = shift augmentation
                off = torch.randint(0, T - L + 1, (B,), device=self.device)
                ar = (off[:, None] + torch.arange(L, device=self.device))[:, None, :].expand(B, C, L)
                x = torch.gather(x, 2, ar) + args.noise * torch.randn(B, C, L, device=self.device)
            else:
                s = (T - L) // 2
                x = x[:, :, s:s + L]
            sig = x
        if vmode == "video":
            vid = self.vid[idx_t].float()
        elif vmode == "videoft":
            Kc = self.frames.shape[1]
            Ku = min(args.ft_frames_use, Kc)
            fsel = np.sort(np.random.choice(Kc, Ku, replace=False)) if train \
                else np.linspace(0, Kc - 1, Ku).round().astype(int)
            x = torch.from_numpy(self.frames[idx[:, None], fsel[None, :]]).to(self.device)   # (B, Ku, 3, S, S)
            if train:                                              # flip + brightness/contrast jitter, per clip
                x = x.float()
                B = x.shape[0]
                flip = (torch.rand(B, device=self.device) < 0.5).view(B, 1, 1, 1, 1)
                x = torch.where(flip, x.flip(-1), x)
                br = (0.8 + 0.4 * torch.rand(B, 1, 1, 1, 1, device=self.device))
                ct = (0.8 + 0.4 * torch.rand(B, 1, 1, 1, 1, device=self.device))
                x = (((x - 128.0) * ct + 128.0) * br).clamp(0, 255)
            vid = x
        return sig, vid, self.y[idx_t]


@torch.no_grad()
def predict(model, data, idx, args, vmode):
    model.eval()
    probs, step = [], (32 if vmode == "videoft" else 256)
    for i in range(0, len(idx), step):
        sig, vid, _ = data.batch(idx[i:i + step], False, args, vmode)
        out, _ = model(sig, vid)
        probs.append(torch.softmax(out["fused"].float(), 1).cpu())
    return torch.cat(probs).numpy()


def fit_bias(logp, y, C, grid=np.linspace(-1.0, 1.0, 21), rounds=2):
    """Per-class additive offsets on log-probabilities that maximise macro-F1 on the validation subjects."""
    b = np.zeros(C)
    best = f1_score(y, (logp + b).argmax(1), average="macro", zero_division=0)
    for _ in range(rounds):
        for c in range(C):
            for v in sorted(grid, key=abs):
                bb = b.copy()
                bb[c] = v
                m = f1_score(y, (logp + bb).argmax(1), average="macro", zero_division=0)
                if m > best + 1e-3:
                    best, b = m, bb
    return b


def metrics(yt, pred, C):
    m = dict(acc=accuracy_score(yt, pred), f1=f1_score(yt, pred, average="macro", zero_division=0))
    if C > 2:
        m["mae"] = float(np.mean(np.abs(yt - pred)))
        m["pm1"] = float(np.mean(np.abs(yt - pred) <= 1))
    return m


def train_one(seed, tag, data, tr_idx, va_idx, te_idx, args, n_classes, mods, vmode):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MultiModalNet(mods, video_dim=data.vid.shape[-1] if data.vid is not None else 768,
                          n_classes=n_classes, dropout=args.dropout, ft_model=args.ft_model,
                          ft_unfreeze=args.ft_unfreeze, ft_pretrained=not args.ft_no_pretrained).to(data.device)
    bb = [p for n, p in model.named_parameters() if p.requires_grad and ".backbone." in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and ".backbone." not in n]
    groups = [dict(params=rest, lr=args.lr)] + ([dict(params=bb, lr=args.ft_lr)] if bb else [])
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = data.device == "cuda" and vmode == "videoft"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    y_np = data.y.cpu().numpy()
    best, best_f1, bad, ep_done = None, -1.0, 0, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot, nb = 0.0, 0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            sig, vid, y = data.batch(perm[i:i + args.batch], True, args, vmode)
            out, _ = model(sig, vid)
            loss = multitask_loss(out, y, n_classes=n_classes,
                                  sigma=args.ordinal_sigma if n_classes > 2 else 0.0,
                                  reg_weight=args.ordinal_reg if n_classes > 2 else 0.0,
                                  label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
        sched.step()
        pv = predict(model, data, va_idx, args, vmode).argmax(1)
        v_acc, v_f1 = accuracy_score(y_np[va_idx], pv), f1_score(y_np[va_idx], pv, average="macro", zero_division=0)
        ep_done = ep
        if v_f1 > best_f1:
            best_f1, best, bad = v_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if ep == 1 or ep % args.log_every == 0 or bad == 0:
            print(f"  {tag} ep {ep:3d}  train_loss {tot / max(nb, 1):.4f}  val_acc {v_acc:.4f}  "
                  f"val_f1 {v_f1:.4f}  {'*' if bad == 0 else ''}")
        if bad >= args.patience:
            print(f"  {tag}: early stop at epoch {ep}")
            break
    model.load_state_dict(best)
    return predict(model, data, va_idx, args, vmode), predict(model, data, te_idx, args, vmode), best_f1


def run_fold(k, data, tr, va, te, args, n_classes, mods, vmode):
    pvs, pts, vf = [], [], []
    for e in range(args.ensemble):
        tag = f"fold {k}" + (f" seed {e + 1}/{args.ensemble}" if args.ensemble > 1 else "")
        pv, pt, f = train_one(args.seed + 1000 * k + 17 * e, tag, data, tr, va, te, args, n_classes, mods, vmode)
        pvs.append(pv)
        pts.append(pt)
        vf.append(f)
    pv, pt = np.mean(pvs, 0), np.mean(pts, 0)                      # ensemble = average of probabilities
    y_np = data.y.cpu().numpy()
    yv, yt = y_np[va], y_np[te]
    bias = fit_bias(np.log(pv + 1e-9), yv, n_classes) if args.calibrate else np.zeros(n_classes)
    pred_raw = pt.argmax(1)
    pred_cal = (np.log(pt + 1e-9) + bias).argmax(1)
    return dict(raw=metrics(yt, pred_raw, n_classes), cal=metrics(yt, pred_cal, n_classes), val_f1=float(np.mean(vf)),
                pred=pred_raw, pred_cal=pred_cal, prob=pt, y=yt, bias=bias)


def summarise(per_fold, y_all, p_all, C):
    df = pd.DataFrame(per_fold)
    s = {}
    for key in df.columns:
        s[f"{key}_mean"] = float(df[key].mean())
        s[f"{key}_std"] = float(df[key].std(ddof=0))
    s["pooled_acc"] = float(accuracy_score(y_all, p_all))
    s["pooled_f1"] = float(f1_score(y_all, p_all, average="macro", zero_division=0))
    if C > 2:
        s["pooled_mae"] = float(np.mean(np.abs(y_all - p_all)))
        s["pooled_pm1"] = float(np.mean(np.abs(y_all - p_all) <= 1))
    return s


def run_task(task, data, folds, args, out_root, mods, vmode):
    cfg = TASKS[task]
    classes, C = cfg["classes"], len(cfg["classes"])
    out = out_root / task
    out.mkdir(parents=True, exist_ok=True)
    data.set_task(task)
    meta, subj = data.meta, data.meta.subject.values
    print(f"\n########## TASK: {task}  ({' / '.join(classes)})  modalities {mods}  chance = {1 / C:.2f} ##########")

    per = {"raw": [], "cal": []}
    cms = {"raw": np.zeros((C, C), dtype=int), "cal": np.zeros((C, C), dtype=int)}
    ys, ps, pcs, preds = [], [], [], []
    for k in range(len(folds)):
        t0 = time.time()
        test_sub = set(folds[k])
        pool = sorted(set(meta.subject) - test_sub)
        rs = np.random.RandomState(args.seed * 100 + k)           # same val subjects for every task / modality
        val_sub = set(rs.choice(pool, min(args.val_subjects, len(pool) - 1), replace=False))
        te = np.where(np.isin(subj, list(test_sub)) & data.mask)[0]
        va = np.where(np.isin(subj, list(val_sub)) & data.mask)[0]
        tr = np.where(~np.isin(subj, list(test_sub | val_sub)) & data.mask)[0]
        assert not (set(subj[tr]) & set(subj[te])) and not (set(subj[va]) & set(subj[te]))
        print(f"\n=== [{task}] fold {k + 1}/{len(folds)}: train {len(tr)} clips / {len(set(subj[tr]))} subjects | "
              f"val {len(va)} / {len(val_sub)} | test {len(te)} / {len(test_sub)} ===")
        r = run_fold(k, data, tr, va, te, args, C, mods, vmode)
        for v, pk in (("raw", "pred"), ("cal", "pred_cal")):
            per[v].append(dict(r[v], fold=k))
            cms[v] += confusion_matrix(r["y"], r[pk], labels=range(C))
        extra = (f"  MAE {r['raw']['mae']:.3f}  ±1-acc {r['raw']['pm1']:.4f}" if C > 2 else "")
        print(f"[{task}] fold {k + 1} TEST  raw acc {r['raw']['acc']:.4f} f1 {r['raw']['f1']:.4f}{extra} | "
              f"cal acc {r['cal']['acc']:.4f} f1 {r['cal']['f1']:.4f}  ({time.time() - t0:.0f}s)")
        ys.append(r["y"])
        ps.append(r["pred"])
        pcs.append(r["pred_cal"])
        df = meta.iloc[te][["subject", "sample"]].copy()
        df["label"], df["pred"], df["pred_cal"], df["fold"] = r["y"], r["pred"], r["pred_cal"], k
        for c in range(C):
            df[f"p_{classes[c]}"] = r["prob"][:, c]
        preds.append(df)

    y_all = np.concatenate(ys)
    summary = dict(task=task, classes=classes, modalities=mods, n_folds=len(folds), ensemble=args.ensemble)
    summary["raw"] = summarise([{k: v for k, v in d.items() if k != "fold"} for d in per["raw"]], y_all, np.concatenate(ps), C)
    summary["cal"] = summarise([{k: v for k, v in d.items() if k != "fold"} for d in per["cal"]], y_all, np.concatenate(pcs), C)
    summary["per_fold_raw"], summary["per_fold_cal"] = per["raw"], per["cal"]
    json.dump(summary, open(out / "results.json", "w"), indent=2)
    pd.concat(preds).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(cms["raw"], index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")
    pd.DataFrame(cms["cal"], index=classes, columns=classes).to_csv(out / "confusion_matrix_cal.csv")

    print(f"\n================ {task} [{'+'.join(mods)}]: subject-independent results ({len(folds)} folds) ================")
    for v, label in (("raw", "RAW argmax"), ("cal", "CALIBRATED (offsets tuned on validation subjects)")):
        s = summary[v]
        line = (f"{label:50s} acc {s['acc_mean']:.4f} +/- {s['acc_std']:.4f} (pooled {s['pooled_acc']:.4f})   "
                f"macro-F1 {s['f1_mean']:.4f} +/- {s['f1_std']:.4f} (pooled {s['pooled_f1']:.4f})")
        if C > 2:
            line += f"   MAE {s['mae_mean']:.3f}  ±1-acc {s['pm1_mean']:.4f}"
        print(line)
    print(f"chance accuracy {1 / C:.2f}.  pooled confusion matrix, raw (rows = true, cols = predicted):")
    print(pd.DataFrame(cms["raw"], index=classes, columns=classes).to_string())
    print(f"saved to {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=r"D:\BioVid\cache")
    ap.add_argument("--out", default=r"D:\BioVid\runs\default")
    ap.add_argument("--tasks", default="5class,binary", help="comma list of: 5class, binary")
    ap.add_argument("--modalities", default="bio",
                    help="runs, comma separated; '+' joins channels. e.g. bio,bio+video,bio+videoft,eda,emg,ecg")
    ap.add_argument("--folds", type=int, default=29)
    ap.add_argument("--val_subjects", type=int, default=9, help="training subjects held out for early stopping")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--ordinal_sigma", type=float, default=0.8, help="Gaussian ordinal label smoothing, 5class (0 = off)")
    ap.add_argument("--ordinal_reg", type=float, default=0.5, help="expected-value Smooth-L1 weight, 5class (0 = off)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--ensemble", type=int, default=1, help="models (seeds) averaged per fold; 3 is a good final setting")
    ap.add_argument("--no_calibrate", dest="calibrate", action="store_false",
                    help="skip validation-tuned class offsets (cal columns then equal raw)")
    ap.add_argument("--crop", type=int, default=1280, help="signal samples per window (256 Hz -> 5 s)")
    ap.add_argument("--noise", type=float, default=0.02)
    # video fine-tuning (modality 'videoft')
    ap.add_argument("--ft_model", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    ap.add_argument("--ft_unfreeze", type=int, default=1, help="how many LAST ConvNeXt stages to train (1 or 2)")
    ap.add_argument("--ft_lr", type=float, default=5e-5, help="learning rate of the unfrozen ConvNeXt stages")
    ap.add_argument("--ft_frames_use", type=int, default=8, help="frames per clip per step (out of the cached ones)")
    ap.add_argument("--ft_no_pretrained", action="store_true", help="random weights (testing only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    runs = parse_runs(args.modalities)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    assert set(tasks) <= set(TASKS), f"tasks: {list(TASKS)}"
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)
    needed = sorted({m for _, mods in runs for m in mods})
    print(f"device={device} runs={[n for n, _ in runs]} tasks={tasks} ensemble={args.ensemble}")

    data = Data(args.cache, needed, device)
    n_sub = data.meta.subject.nunique()
    if args.folds > n_sub:
        raise SystemExit(f"--folds {args.folds} is larger than the number of subjects ({n_sub})")
    folds = make_folds(data.meta.subject.unique(), args.folds, args.seed)
    pd.DataFrame([(s, k) for k, f in enumerate(folds) for s in f], columns=["subject", "fold"]) \
        .to_csv(out / "fold_assignment.csv", index=False)
    print(f"{len(data.meta)} clips, {n_sub} subjects, {args.folds} folds of "
          f"{sorted(set(len(f) for f in folds))} test subjects each, {args.val_subjects} validation subjects")

    allsum = {}
    for name, mods in runs:
        print(f"\n{'=' * 30} RUN: {name} ({mods}) {'=' * 30}")
        vmode = "videoft" if "videoft" in mods else ("video" if "video" in mods else None)
        exp_out = out / name if len(runs) > 1 else out
        exp_out.mkdir(parents=True, exist_ok=True)
        allsum[name] = [run_task(t, data, folds, args, exp_out, mods, vmode) for t in tasks]

    print("\n" + "=" * 110)
    print(f"{'SUMMARY  (mean +/- std over folds | pooled)':^110}")
    print("=" * 110)
    rec = []
    for t in tasks:
        print(f"\nTask: {t}")
        print(f"  {'run':<16}{'raw acc':<22}{'raw F1':<22}{'cal acc':<22}{'cal F1':<22}" + ("MAE / ±1-acc" if t == "5class" else ""))
        for name, _ in runs:
            s = next(x for x in allsum[name] if x["task"] == t)
            f = lambda d, a, b: f"{d[a + '_mean']:.4f}±{d[a + '_std']:.3f}|{d[b]:.4f}"
            row = f"  {name:<16}{f(s['raw'], 'acc', 'pooled_acc'):<22}{f(s['raw'], 'f1', 'pooled_f1'):<22}" \
                  f"{f(s['cal'], 'acc', 'pooled_acc'):<22}{f(s['cal'], 'f1', 'pooled_f1'):<22}"
            if t == "5class":
                row += f"{s['raw']['mae_mean']:.3f} / {s['raw']['pm1_mean']:.4f}"
            print(row)
            rec.append(dict(task=t, run=name, raw_acc=s["raw"]["acc_mean"], raw_f1=s["raw"]["f1_mean"],
                            cal_acc=s["cal"]["acc_mean"], cal_f1=s["cal"]["f1_mean"],
                            raw_acc_pooled=s["raw"]["pooled_acc"], raw_f1_pooled=s["raw"]["pooled_f1"]))
    pd.DataFrame(rec).to_csv(out / "summary.csv", index=False)
    print(f"\nsaved {out / 'summary.csv'}")


if __name__ == "__main__":
    main()