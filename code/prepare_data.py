"""
Stage 1 - build the cache that train_cv.py reads.

  1. scan BioVid Part A and build meta.csv (subject, sample, label, file paths)
  2. biosignals (EDA, ECG, trapezius EMG): load -> (optional Thiam filters) ->
     resample to 256 Hz -> ECG polynomial detrend -> per-subject z-score
     -> signals.npy   shape (N, 3, T)
  3. video: 38 evenly spaced frames per clip -> face crop -> frozen ConvNeXt V2
     -> embeddings.npy   shape (N, 38, D)   (float16)
  4. (optional, step `frames`) a small cache of face-crop FRAMES, used to FINE-TUNE ConvNeXt V2 in train_cv.py
     -> frames.npy       shape (N, ft_frames, 3, ft_size, ft_size)   (uint8)

Run  `python prepare_data.py --root <PartA folder> --check`  first to confirm the
script finds your files and reads them correctly.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

LABELS = {"BL1": 0, "PA1": 1, "PA2": 2, "PA3": 3, "PA4": 4}
CHANNELS = ["eda", "ecg", "emg"]
LABEL_RE = re.compile(r"-(BL1|PA[1-4])-")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


# --------------------------------------------------------------------------- #
# 1. index
# --------------------------------------------------------------------------- #
def find_dir(root: Path, names):
    """First sub-folder of `root` (any depth) whose name is in `names`."""
    for n in names:
        hits = sorted(p for p in root.rglob(n) if p.is_dir())
        if hits:
            return hits[0]
    return None


def build_meta(bio_dir: Path, video_dir: Path) -> pd.DataFrame:
    bios = {p.name[: -len("_bio.csv")]: p for p in bio_dir.rglob("*_bio.csv")}
    vids = {p.stem.replace("_video", ""): p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXT}
    print(f"[meta] biosignal files: {len(bios)} | video files: {len(vids)}")
    both = sorted(set(bios) & set(vids))
    if len(both) < len(bios) or len(both) < len(vids):
        print(f"[meta] WARNING: only {len(both)} samples have both a biosignal file and a video "
              f"({len(bios) - len(both)} bio-only, {len(vids) - len(both)} video-only)")
    rows = []
    for key in both:
        m = LABEL_RE.search(key)
        if not m:
            print(f"[meta] skipping '{key}': cannot parse class (BL1/PA1..PA4) from name")
            continue
        rows.append(dict(subject=key.split("-")[0], sample=key, label=LABELS[m.group(1)],
                         class_name=m.group(1), bio_path=str(bios[key]), video_path=str(vids[key])))
    if not rows:
        sys.exit("[meta] no matching biosignal/video pairs found. Check --root / --bio_dir / --video_dir.")
    meta = pd.DataFrame(rows).sort_values(["subject", "sample"]).reset_index(drop=True)
    return meta


# --------------------------------------------------------------------------- #
# 2. biosignals
# --------------------------------------------------------------------------- #
def load_bio(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]

    def pick(*keys):
        for k in keys:
            for c in df.columns:
                if k in c:
                    return df[c].to_numpy(dtype=np.float64)
        raise KeyError(f"none of {keys} found in columns {list(df.columns)} of {path}")

    return pick("gsr", "eda"), pick("ecg"), pick("trapez", "emg")


def thiam_filters(eda, ecg, emg, fs):
    """Filters from Thiam et al. 2019. Applied at the ORIGINAL sampling rate.
    BioVid's 'biosignals_filtered' files are already filtered, so this is off by default."""
    eda = sosfiltfilt(butter(3, 0.2, "low", fs=fs, output="sos"), eda)
    emg = sosfiltfilt(butter(4, [20, 250], "band", fs=fs, output="sos"), emg)
    ecg = sosfiltfilt(butter(3, [0.1, 250], "band", fs=fs, output="sos"), ecg)
    return eda, ecg, emg


def fix_len(x, n):
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)), mode="edge")


def detrend_poly(x, deg=5):
    t = np.linspace(-1.0, 1.0, len(x))
    return x - np.polyval(np.polyfit(t, x, deg), t)


def process_signals(meta, fs, target_fs, apply_filters, seconds=5.5):
    from tqdm import tqdm

    g = math.gcd(int(fs), int(target_fs))
    up, down = int(target_fs) // g, int(fs) // g
    T = int(round(seconds * target_fs))
    out = np.zeros((len(meta), 3, T), dtype=np.float32)
    lens = []
    for i, p in enumerate(tqdm(meta.bio_path, desc="biosignals")):
        eda, ecg, emg = load_bio(p)
        lens.append(len(eda))
        if apply_filters:
            eda, ecg, emg = thiam_filters(eda, ecg, emg, fs)
        chans = []
        for k, x in enumerate((eda, ecg, emg)):
            x = np.nan_to_num(x)
            x = resample_poly(x, up, down) if up != down else x
            x = fix_len(x, T)
            if k == 1:                      # ECG: remove baseline wander (5th degree polynomial)
                x = detrend_poly(x, 5)
            chans.append(x)
        out[i] = np.stack(chans)
    print(f"[signals] raw length per clip: min {min(lens)}, max {max(lens)} samples "
          f"(expected ~{int(seconds * fs)} at {fs} Hz) -> resampled to {T} samples at {target_fs} Hz")

    # per-subject z-score, per channel (label-free, uses only that subject's own recordings)
    for s in meta.subject.unique():
        idx = np.where(meta.subject.values == s)[0]
        blk = out[idx]                                   # (n, 3, T)
        mu = blk.mean(axis=(0, 2), keepdims=True)
        sd = blk.std(axis=(0, 2), keepdims=True) + 1e-8
        out[idx] = (blk - mu) / sd
    return out


# --------------------------------------------------------------------------- #
# 3. video -> frozen ConvNeXt V2 embeddings
# --------------------------------------------------------------------------- #
class ClipDataset:
    """Returns (K,3,S,S) uint8 face crops for one clip + whether a face was detected."""

    def __init__(self, paths, n_frames, size):
        self.paths, self.K, self.size = list(paths), n_frames, size
        self._det = None

    def __len__(self):
        return len(self.paths)

    def _detector(self):
        if self._det is None:
            try:
                det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                self._det = det if not det.empty() else False
            except Exception:
                self._det = False                      # OpenCV build without Haar cascades
        return self._det

    def _read_frames(self, path):
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            n = 138
        want = np.linspace(0, n - 1, self.K).round().astype(int)
        want_set = set(want.tolist())
        frames, i = [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in want_set:
                frames.append(f)
            i += 1
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames could be read from {path}")
        while len(frames) < self.K:                      # frame count was off: repeat last frame
            frames.append(frames[-1])
        return frames[: self.K]

    def _face_box(self, frames):
        det = self._detector()
        K = len(frames)
        for j in ((K // 2, 0, K - 1, K // 4, 3 * K // 4) if det else ()):
            gray = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
            faces = det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces):
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                return x + w / 2, y + h / 2, 1.4 * max(w, h), True
        H, W = frames[0].shape[:2]
        return W / 2, H / 2, float(min(H, W)), False

    def __getitem__(self, i):
        import torch

        frames = self._read_frames(self.paths[i])
        cx, cy, side, found = self._face_box(frames)
        H, W = frames[0].shape[:2]
        side = min(side, H, W)
        x0 = int(round(min(max(cx - side / 2, 0), W - side)))
        y0 = int(round(min(max(cy - side / 2, 0), H - side)))
        s = int(round(side))
        out = np.empty((self.K, self.size, self.size, 3), dtype=np.uint8)
        for k, f in enumerate(frames):
            crop = f[y0:y0 + s, x0:x0 + s]
            crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
            out[k] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous(), int(found), i


def extract_embeddings(meta, cache, args):
    import timm
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[video] device={dev} model={args.model} pretrained={not args.no_pretrained}")
    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=0).to(dev).eval()
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device=dev).view(1, 3, 1, 1) * 255
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device=dev).view(1, 3, 1, 1) * 255
    D, N, K = int(model.num_features), len(meta), args.n_frames

    emb_path, done_path = cache / "embeddings.npy", cache / "emb_done.npy"
    info = dict(model=args.model, n_frames=K, dim=D, img_size=args.img_size, pretrained=not args.no_pretrained)
    if emb_path.exists() and done_path.exists() and (cache / "emb_info.json").exists() \
            and json.load(open(cache / "emb_info.json")) == info:
        emb = np.lib.format.open_memmap(emb_path, mode="r+")
        done = np.load(done_path)
        print(f"[video] resuming: {int(done.sum())}/{N} clips already done")
    else:
        emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(N, K, D))
        done = np.zeros(N, dtype=bool)
        json.dump(info, open(cache / "emb_info.json", "w"))

    todo = np.where(~done)[0]
    if len(todo):
        ds = ClipDataset(meta.video_path.values, K, args.img_size)
        loader = DataLoader(Subset(ds, todo.tolist()), batch_size=args.batch_clips, shuffle=False,
                            num_workers=args.workers, pin_memory=(dev == "cuda"))
        fallbacks, seen = 0, 0
        with torch.no_grad():
            for b, (x, found, idx) in enumerate(tqdm(loader, desc="video embeddings")):
                B = x.shape[0]
                x = x.to(dev, non_blocking=True).view(B * K, 3, args.img_size, args.img_size).float()
                x = (x - mean) / std
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                    f = model(x)
                emb[idx.numpy()] = f.float().view(B, K, D).cpu().numpy().astype(np.float16)
                done[idx.numpy()] = True
                fallbacks += int((found == 0).sum())
                seen += B
                if (b + 1) % 50 == 0:
                    emb.flush()
                    np.save(done_path, done)
        emb.flush()
        np.save(done_path, done)
        print(f"[video] face detector found a face in {100 * (1 - fallbacks / max(seen, 1)):.1f}% of clips "
              f"(others use a centre crop)")
    print(f"[video] embeddings: {emb_path}  shape={emb.shape}")


def extract_frames(meta, cache, args):
    """Cache small face-crop frames (uint8) so ConvNeXt V2 can be fine-tuned without re-decoding videos."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    N, K, S = len(meta), args.ft_frames, args.ft_size
    fr_path, done_path, info_path = cache / "frames.npy", cache / "frames_done.npy", cache / "frames_info.json"
    info = dict(n_frames=K, size=S)
    if fr_path.exists() and done_path.exists() and info_path.exists() and json.load(open(info_path)) == info:
        fr = np.lib.format.open_memmap(fr_path, mode="r+")
        done = np.load(done_path)
        print(f"[frames] resuming: {int(done.sum())}/{N} clips already done")
    else:
        fr = np.lib.format.open_memmap(fr_path, mode="w+", dtype=np.uint8, shape=(N, K, 3, S, S))
        done = np.zeros(N, dtype=bool)
        json.dump(info, open(info_path, "w"))
    todo = np.where(~done)[0]
    if len(todo):
        ds = ClipDataset(meta.video_path.values, K, S)
        loader = DataLoader(Subset(ds, todo.tolist()), batch_size=8, shuffle=False, num_workers=args.workers)
        for b, (x, found, idx) in enumerate(tqdm(loader, desc="frames")):
            fr[idx.numpy()] = x.numpy()
            done[idx.numpy()] = True
            if (b + 1) % 50 == 0:
                fr.flush()
                np.save(done_path, done)
        fr.flush()
        np.save(done_path, done)
    print(f"[frames] {fr_path}  shape={fr.shape}  (~{fr.nbytes / 1e9:.1f} GB)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="BioVid Part A folder (contains video/ and biosignals_*/); "
                         "not needed if both --bio_dir and --video_dir are given")
    ap.add_argument("--bio_dir", default=r"D:\BioVid\BioVid Part A\biosignals_filtered\biosignals_filtered",
                    help="biosignal folder")
    ap.add_argument("--video_dir", default=r"D:\BioVid\BioVid Part A\video", help="video folder")
    ap.add_argument("--cache", default=r"D:\BioVid\cache")
    ap.add_argument("--steps", default="meta,signals,video",
                    help="comma list of: meta, signals, video (cached embeddings), frames (for fine-tuning)")
    ap.add_argument("--check", action="store_true", help="only inspect the dataset, then exit")
    ap.add_argument("--fs", type=int, default=512, help="original biosignal sampling rate")
    ap.add_argument("--target_fs", type=int, default=256)
    ap.add_argument("--apply_filters", action="store_true",
                    help="apply Thiam filters (use with biosignals_raw; filtered files are already filtered)")
    ap.add_argument("--n_frames", type=int, default=42)
    ap.add_argument("--img_size", type=int, default=288)
    ap.add_argument("--model", default="convnextv2_base.fcmae_ft_in22k_in1k")
    ap.add_argument("--no_pretrained", action="store_true", help="random weights (testing only)")
    ap.add_argument("--batch_clips", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ft_frames", type=int, default=16, help="frames per clip cached for fine-tuning (step 'frames')")
    ap.add_argument("--ft_size", type=int, default=128, help="face-crop size cached for fine-tuning (step 'frames')")
    args = ap.parse_args()

    m_str = args.model.lower().strip()
    if "base" in m_str:
        args.model = "convnextv2_base.fcmae_ft_in22k_in1k"
    elif "tiny" in m_str:
        args.model = "convnextv2_tiny.fcmae_ft_in22k_in1k"

    if not args.root and not (args.bio_dir and args.video_dir):
        ap.error("give --root, or give both --bio_dir and --video_dir")
    root, cache = (Path(args.root) if args.root else None), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    bio_dir = Path(args.bio_dir) if args.bio_dir else (
        find_dir(root, ["biosignals_filtered", "biosignals_raw"]) or root)
    video_dir = Path(args.video_dir) if args.video_dir else (find_dir(root, ["video", "videos"]) or root)
    print(f"[paths] biosignals: {bio_dir}\n[paths] video:      {video_dir}")

    steps = set(args.steps.split(","))
    meta_path = cache / "meta.csv"
    if "meta" in steps or not meta_path.exists():
        meta = build_meta(bio_dir, video_dir)
        meta.to_csv(meta_path, index=False)
    meta = pd.read_csv(meta_path)
    print(f"[meta] {len(meta)} clips | {meta.subject.nunique()} subjects | "
          f"class counts: {meta.class_name.value_counts().sort_index().to_dict()}")

    if args.check:
        eda, ecg, emg = load_bio(meta.bio_path[0])
        print(f"[check] first biosignal file {meta.bio_path[0]}\n        samples: {len(eda)} "
              f"(5.5 s at {args.fs} Hz would be {int(5.5 * args.fs)})")
        cap = cv2.VideoCapture(meta.video_path[0])
        print(f"[check] first video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames, "
              f"{cap.get(cv2.CAP_PROP_FPS):.1f} fps, "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return

    if "signals" in steps:
        sig = process_signals(meta, args.fs, args.target_fs, args.apply_filters)
        np.save(cache / "signals.npy", sig)
        print(f"[signals] saved {cache / 'signals.npy'} shape={sig.shape}")
    if "video" in steps:
        extract_embeddings(meta, cache, args)
    if "frames" in steps:
        extract_frames(meta, cache, args)


if __name__ == "__main__":
    main()