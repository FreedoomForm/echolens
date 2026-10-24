#!/usr/bin/env python3
"""One-shot accent events for EchoLens: short (0.9 s) punchy material-matched
sounds rendered from FSD50K clips. Stage output: data/stage_events/<MATERIAL>/.

Run AFTER fsd_pipeline.py download (reuses its helpers), BEFORE select.
"""
import csv, json, os, sys, threading, time
import concurrent.futures as cf
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsd_pipeline as fp                     # noqa: E402
import fsd_build_candidates as F              # noqa: E402

DATA = "/home/z/my-project/data"
STAGE_EV = os.path.join(DATA, "stage_events")
WORKERS = 10
PER_MAT = 10
PICK_EVENT = 4

def main():
    os.makedirs(STAGE_EV, exist_ok=True)
    split = {}
    for fn, s in (("eval.csv", "eval"), ("dev.csv", "dev")):
        with open(os.path.join(DATA, fn), newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if row and row[0]:
                    split[row[0]] = s
    info = {}
    for fn in ("eval_clips_info.json", "dev_clips_info.json"):
        info.update(json.load(open(os.path.join(DATA, fn))))

    by_label = {}
    for fn in ("eval.csv", "dev.csv"):
        with open(os.path.join(DATA, fn), newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            fi = 0
            for row in r:
                if not row or not row[0]:
                    continue
                for lb in row[1].split(","):
                    by_label.setdefault(lb, []).append(row[0])

    vocab = set()
    with open(os.path.join(DATA, "vocabulary.csv"), newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                vocab.add(row[1])

    tasks = []
    for mat, (_lb, _tg, ev_labels) in F.MATERIALS.items():
        out_dir = os.path.join(STAGE_EV, mat)
        have = {f[:-4] for f in os.listdir(out_dir) if f.endswith(".ogg")} if os.path.isdir(out_dir) else set()
        picked = 0
        for cid in by_label.get(ev_labels[0], []) if ev_labels else []:
            if picked >= PER_MAT:
                break
            if cid in have:
                picked += 1
                continue
            meta = info.get(cid)
            if not meta:
                continue
            lic = F.norm_lic(meta.get("license", ""))
            if lic not in ("CC0", "CC-BY"):
                continue
            tasks.append((mat, cid, ev_labels))
            picked += 1
    print("event tasks:", len(tasks))

    done = [0]
    lock = threading.Lock()

    def work(t):
        mat, cid, ev_labels = t
        out_dir = os.path.join(STAGE_EV, mat)
        os.makedirs(out_dir, exist_ok=True)
        ogg = os.path.join(out_dir, f"{cid}.ogg")
        meta_path = ogg + ".json"
        ok = False
        if not (os.path.exists(ogg) and os.path.exists(meta_path)):
            wav = ogg + ".tmp.wav"
            met = {"why": "dl"}
            if fp.download_clip(cid, split[cid], wav):
                try:
                    x, sr = sf.read(wav, dtype="float64", always_2d=False)
                    if x.ndim > 1:
                        x = x.mean(axis=1)
                    oksc, score, met = fp.score_clip(x, sr)
                    if oksc:
                        seg = fp.pick_segment(x, sr, fp.EVENT_LEN, loop=False)
                        if seg:
                            s0, n = seg
                            y = fp.fades(x[s0:s0 + n], sr, ms=6)
                            y = fp.normalize(y, target_rms=0.14)
                            met["promo"] = 0.6
                            met["score"] = round(score + 2.0, 1)
                            ok = fp.encode(y, sr, ogg)
                except Exception as e:
                    met = {"why": f"err:{type(e).__name__}"}
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
            if done[0] % 50 == 0:
                print(f"  {done[0]}/{len(tasks)}")

    t0 = time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, tasks))
    print(f"events done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
