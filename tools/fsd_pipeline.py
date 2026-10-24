#!/usr/bin/env python3
"""EchoLens sound pack pipeline: download FSD50K clips, score quality, curate
the cleanest segments per class/material, render seamless loops + one-shot
events, encode OGG, and stage everything for the repo assets.

Stages (run sequentially):
  download  - thread pool downloads candidate clips and processes each one
              into a scored, loopified OGG in the staging area
  select    - rank staged clips per target, copy winners to the repo assets,
              write manifest.json + ATTRIBUTIONS.md

Usage: python3 fsd_pipeline.py [download|select|all]
"""
import csv, json, math, os, shutil, subprocess, sys, tempfile, threading, time
import concurrent.futures as cf
import numpy as np
import soundfile as sf

DATA = "/home/z/my-project/data"
STAGE = os.path.join(DATA, "stage")
REPO = "/home/z/my-project/echolens"
ASSETS = os.path.join(REPO, "app/src/main/assets/sounds")
HF = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips"

LOOP_TARGETS = {  # desired loop length (s) per material bed
    "ENGINE": 4.5, "WATER": 4.5, "PLANT": 4.5, "ELECTRONICS": 4.0,
    "NEUTRAL": 4.0, "PERSON": 3.5, "FABRIC": 3.0, "PAPER": 3.0,
}
EVENT_LEN = 0.9          # one-shot accent length (s)
PICK_CLASS = 8           # loops per class
PICK_MAT = 10            # loops per material
PICK_EVENT = 4           # one-shots per material
WORKERS = 10
SR_OUT = 48000

_stop = threading.Event()

# ------------------------------------------------------------------ helpers

def sh(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, timeout=timeout)

def download_clip(cid: str, split: str, dst: str) -> bool:
    url = f"{HF}/{split}/{cid}.wav"
    for attempt in range(3):
        if _stop.is_set():
            return False
        r = sh(["curl", "-sL", "--max-time", "90", "-o", dst, url])
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return True
        time.sleep(1.5 * (attempt + 1))
    return False

def frame_rms(x, sr, hop_s=0.020):
    hop = max(1, int(hop_s * sr))
    n = len(x) // hop
    if n == 0:
        return np.zeros(1)
    trim = x[: n * hop].reshape(n, hop)
    return np.sqrt((trim ** 2).mean(axis=1)) + 1e-12

def pick_segment(x, sr, want_s, loop: bool):
    """Pick the most stable loud window; returns (start, n) or None."""
    N = len(x)
    want = int(want_s * sr)
    if loop:
        need = int((want_s + 0.6) * sr)
        if N < int(want_s * 0.75) :
            return None
        if N < need:
            want = int(N * 0.7)
            start = 0
            return start, want
        rms = frame_rms(x, sr, 0.02)
        hop = int(0.02 * sr)
        w = max(2, int(want / hop))
        if w >= len(rms):
            return 0, want
        c = np.cumsum(np.insert(rms, 0, 0))
        c2 = np.cumsum(np.insert(rms ** 2, 0, 0))
        idx = np.arange(0, len(rms) - w, max(1, w // 8))
        mean = (c[idx + w] - c[idx]) / w
        var = (c2[idx + w] - c2[idx]) / w - mean ** 2
        stab = mean / (np.sqrt(np.maximum(var, 0.0)) + 1e-9)
        best = idx[int(np.argmax(stab * np.log1p(mean * 50)))]
        return int(best * hop), want
    else:  # one-shot: loudest onset region
        need = int(EVENT_LEN * sr)
        if N < need:
            if N < 0.35 * sr:
                return None
            return 0, N
        rms = frame_rms(x, sr, 0.01)
        hop = int(0.01 * sr)
        w = max(2, int(need / hop))
        if w >= len(rms):
            return 0, need
        c = np.cumsum(np.insert(rms, 0, 0))
        idx = np.arange(0, len(rms) - w, max(1, w // 6))
        mean = (c[idx + w] - c[idx]) / w
        best = idx[int(np.argmax(mean))]
        return int(best * hop), need

def score_clip(x, sr):
    """Quality gate + score. Returns (ok, score, metrics)."""
    N = len(x)
    dur = N / sr
    if dur < 0.7 or dur > 60:
        return False, 0, {"why": "dur"}
    peak = float(np.max(np.abs(x)))
    if peak < 0.03:
        return False, 0, {"why": "quiet"}
    clip_frac = float(np.mean(np.abs(x) > 0.985))
    if clip_frac > 0.02:
        return False, 0, {"why": "clip"}
    rms = frame_rms(x, sr, 0.02)
    loud = np.mean(np.sort(rms)[-max(1, len(rms) // 5):])
    quiet = np.percentile(rms, 10)
    snr = 20 * math.log10(loud / quiet)
    if snr < 9:
        return False, 0, {"why": "snr", "snr": snr}
    crest = 20 * math.log10(peak / loud)
    return True, snr, {"snr": round(snr, 1), "crest": round(crest, 1),
                       "dur": round(dur, 1), "peak": round(peak, 2)}

def loopify(seg, sr, xfade_s=0.08):
    """Make a seamless loop via tail->head overlap-add."""
    xf = int(xfade_s * sr)
    if xf <= 0 or len(seg) <= 2 * xf:
        return seg
    y = seg[:-xf].copy()
    t = seg[-xf:]
    fade = np.linspace(0, 1, xf, dtype=np.float64)
    y[-xf:] = y[-xf:] * (1 - fade) + t * fade
    return y

def fades(y, sr, ms=8):
    n = int(ms / 1000 * sr)
    if 2 * n < len(y):
        env = np.ones_like(y)
        env[:n] = np.linspace(0, 1, n)
        env[-n:] = np.linspace(1, 0, n)
        return y * env
    return y

def normalize(y, target_rms=0.112):
    r = math.sqrt((y ** 2).mean()) + 1e-9
    y = y * (target_rms / r)
    p = np.max(np.abs(y))
    if p > 0.97:
        y = y * (0.97 / p)
    return np.clip(y, -1, 1)

def encode(y, sr, ogg_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, y.astype(np.float32), sr, subtype="FLOAT")
        r = sh(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                "-ar", str(SR_OUT), "-ac", "1",
                "-c:a", "libvorbis", "-q:a", "3", ogg_path], 60)
        return r.returncode == 0 and os.path.exists(ogg_path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

# ------------------------------------------------------------------ stages

def load_inputs():
    lt = json.load(open(os.path.join(DATA, "labeltable.json")))
    cands = json.load(open(os.path.join(DATA, "candidates.json")))
    info = {}
    for fn in ("eval_clips_info.json", "dev_clips_info.json"):
        info.update(json.load(open(os.path.join(DATA, fn))))
    split = {}
    for fn, s in (("eval.csv", "eval"), ("dev.csv", "dev")):
        with open(os.path.join(DATA, fn), newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0]:
                    split[row[0]] = s
    return lt, cands, info, split

def stage_download():
    lt, cands, info, split = load_inputs()
    os.makedirs(STAGE, exist_ok=True)
    tasks = []
    for target, tc in cands.items():
        kind = tc["kind"]
        for c in tc["clips"]:
            cid = c["id"]
            if cid not in split:
                continue
            tasks.append((target, kind, cid, c.get("promo", 0.5)))
    print(f"tasks: {len(tasks)}")
    done = [0]
    lock = threading.Lock()

    def work(t):
        target, kind, cid, promo = t
        out_dir = os.path.join(STAGE, target.replace(" ", "_").replace("/", "_"))
        ogg = os.path.join(out_dir, f"{cid}.ogg")
        meta_path = ogg + ".json"
        if os.path.exists(ogg) and os.path.exists(meta_path):
            pass
        else:
            os.makedirs(out_dir, exist_ok=True)
            wav = ogg + ".tmp.wav"
            ok_dl = download_clip(cid, split[cid], wav)
            ok = False
            if ok_dl:
                try:
                    xy, sr = sf.read(wav, dtype="float64", always_2d=False)
                    if xy.ndim > 1:
                        xy = xy.mean(axis=1)
                    is_event = False  # decided at select stage; keep loop version
                    want = LOOP_TARGETS.get(target, 3.0)
                    oksc, score, met = score_clip(xy, sr)
                    if oksc:
                        seg = pick_segment(xy, sr, want, loop=True)
                        if seg:
                            s0, n = seg
                            y = xy[s0:s0 + n]
                            y = loopify(y, sr)
                            y = fades(y, sr)
                            y = normalize(y)
                            met["promo"] = promo
                            met["score"] = round(score + 4 * promo, 1)
                            ok = encode(y, sr, ogg)
                except Exception as e:
                    met = {"why": f"err:{type(e).__name__}"}
                else:
                    pass
                finally:
                    try:
                        os.unlink(wav)
                    except OSError:
                        pass
            if ok:
                json.dump(met, open(meta_path, "w"))
            else:
                for p in (ogg, ogg + ".tmp.wav"):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{len(tasks)}")

    t0 = time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, tasks))
    print(f"download+process done in {time.time()-t0:.0f}s")

def stage_select():
    lt, cands, info, split = load_inputs()
    # wipe old synthesized packs (restorable from git history)
    for sub in ("class", "loops", "events"):
        p = os.path.join(ASSETS, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)
    EN, RU, MAT = lt["EN"], lt["RU"], lt["MAT"]
    manifest = {"v": 3, "loops": {}, "events": {}, "class": {}}
    attrib = []
    total_files = 0

    # ---------------- materials ----------------
    for mat, tc in cands.items():
        if tc["kind"] != "material":
            continue
        d = os.path.join(STAGE, mat)
        if not os.path.isdir(d):
            continue
        items = []
        for fn in os.listdir(d):
            if fn.endswith(".ogg"):
                met = json.load(open(os.path.join(d, fn + ".json")))
                items.append((met["score"], fn))
        items.sort(reverse=True)
        loops = [fn for _, fn in items[:PICK_MAT]]
        # one-shot accents come from the dedicated event stage
        ev_dir = os.path.join(DATA, "stage_events", mat)
        events = []
        if os.path.isdir(ev_dir):
            ev_items = []
            for fn in os.listdir(ev_dir):
                if fn.endswith(".ogg"):
                    met = json.load(open(os.path.join(ev_dir, fn + ".json")))
                    ev_items.append((met["score"], fn))
            ev_items.sort(reverse=True)
            events = [fn for _, fn in ev_items[:PICK_EVENT]]
        loops.sort()
        events.sort()
        lpaths, epaths = [], []
        for i, fn in enumerate(loops):
            src = os.path.join(d, fn)
            dst = os.path.join(ASSETS, "loops", mat, f"{mat.lower()}_{i:02d}.ogg")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
            os.unlink(os.path.join(d, fn + ".json"))
            lpaths.append(f"loops/{mat}/{mat.lower()}_{i:02d}.ogg")
            cid = fn[:-4]
            meta = info.get(cid, {})
            attrib.append(f"| loops/{mat}/{mat.lower()}_{i:02d}.ogg | {mat} | {cid} | "
                          f"{meta.get('title','')} | {meta.get('uploader','')} | "
                          f"{meta.get('license','')} |")
            total_files += 1
        for i, fn in enumerate(events):
            src = os.path.join(ev_dir, fn)
            dst = os.path.join(ASSETS, "events", mat, f"{mat.lower()}_e{i}.ogg")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
            os.unlink(os.path.join(ev_dir, fn + ".json"))
            epaths.append(f"events/{mat}/{mat.lower()}_e{i}.ogg")
            total_files += 1
        if lpaths:
            manifest["loops"][mat] = lpaths
        if epaths:
            manifest["events"][mat] = epaths

    # ---------------- classes ----------------
    class_dirs = set()
    for target, tc in cands.items():
        if tc["kind"] != "class":
            continue
        d = os.path.join(STAGE, target.replace(" ", "_").replace("/", "_"))
        if not os.path.isdir(d):
            continue
        items = []
        for fn in os.listdir(d):
            if fn.endswith(".ogg"):
                met = json.load(open(os.path.join(d, fn + ".json")))
                items.append((met["score"], fn))
        if not items:
            continue
        items.sort(reverse=True)
        picked = [fn for _, fn in items[:PICK_CLASS]]
        picked.sort()
        dir_name = target.replace(" ", "_").replace("/", "_")
        class_dirs.add(dir_name)
        paths = []
        for i, fn in enumerate(picked):
            src = os.path.join(d, fn)
            dst = os.path.join(ASSETS, "class", dir_name, f"{dir_name}_{i:02d}.ogg")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
            os.unlink(os.path.join(d, fn + ".json"))
            paths.append(f"class/{dir_name}/{dir_name}_{i:02d}.ogg")
            cid = fn[:-4]
            meta = info.get(cid, {})
            attrib.append(f"| class/{dir_name}/{dir_name}_{i:02d}.ogg | {target} | {cid} | "
                          f"{meta.get('title','')} | {meta.get('uploader','')} | "
                          f"{meta.get('license','')} |")
            total_files += 1
        manifest["class"][target] = {"material": tc["material"], "loops": paths}

    json.dump(manifest, open(os.path.join(ASSETS, "manifest.json"), "w"),
              ensure_ascii=False, indent=1)
    hdr = ("# Sound attributions (FSD50K subset)\n\n"
           "All clips originate from Freesound via the FSD50K dataset "
           "(Fonseca et al., 2020, Zenodo doi:10.5281/zenodo.4060432), "
           "licensed CC0 or CC-BY by their authors.\n\n"
           "| File | Target | FSD50K id | Title | Uploader | License |\n"
           "|------|--------|-----------|-------|----------|---------|\n")
    open(os.path.join(REPO, "app/src/main/assets/sounds/ATTRIBUTIONS.md"), "w").write(hdr + "\n".join(attrib) + "\n")

    nclass = len(manifest["class"])
    nloops = sum(len(v) for v in manifest["loops"].values())
    nev = sum(len(v) for v in manifest["events"].values())
    print(f"classes with packs: {nclass}, material loops: {nloops}, events: {nev}, total files: {total_files}")
    # staging leftovers
    left = sum(len(files) for _, _, files in os.walk(STAGE))
    print("staging leftovers:", left)

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("download", "all"):
        stage_download()
    if what in ("select", "all"):
        stage_select()
