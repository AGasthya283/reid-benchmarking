"""reid-benchmarking - Streamlit testing platform.

    streamlit run app/streamlit_app.py

Pages
  Leaderboard   benchmark results (TrackEval) for every tracker x ReID run, speed, per-sequence view
  Video viewer  side-by-side playback of two runs on a sequence (ID colours, trails, ID switches vs GT)
  Live replay   a MOT17 sequence streamed as a simulated camera feed, tracked online frame by frame
  Test bench    change tracker hyper-parameters / ReID model, re-run on chosen sequences, get
                HOTA / IDF1 / IDSW against GT in seconds, compared with the benchmark default
  Test videos   run the full detector -> ReID -> tracker chain on a simulated test video or an
                uploaded clip and download the annotated result
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reidbench.data import load_sequences, write_eval_gt  # noqa: E402
from reidbench.detect import load_dets  # noqa: E402
from reidbench.viz import (by_frame, draw_frame, id_switch_frames, load_gt, load_mot,  # noqa: E402
                           render_sequence_video)

st.set_page_config(page_title="ReID Benchmark Lab", page_icon="🎯", layout="wide")

TRACKER_NAMES = {"deepocsort": "Deep OC-SORT", "tracktrack": "TrackTrack", "fasttracker": "FastTracker"}
# categorical slots 1-4 (validated palette), neutral grey for the no-ReID baseline
REID_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NONE_COLOR = "#8c8c86"
NO_REID = "none"


# ------------------------------------------------------------------------------------------ data
@st.cache_data
def load_cfg():
    return yaml.safe_load(open(ROOT / "configs" / "benchmark.yaml"))


CFG = load_cfg()
DET = CFG["detector"]["name"]
RUN_DIR = ROOT / "results" / f"{CFG['dataset']['name']}__{DET}"
REID_KEYS = list(CFG["reid_models"])
COLOR_OF = {k: REID_COLORS[i % 4] for i, k in enumerate(REID_KEYS)} | {NO_REID: NONE_COLOR}


@st.cache_resource
def sequences():
    return {s.name: s for s in load_sequences(ROOT / CFG["dataset"]["split_dir"])}


@st.cache_data
def gt_of(seq_name):
    return load_gt(sequences()[seq_name].root)


def det_path(seq):
    return ROOT / "cache" / "dets" / DET / f"{seq}.npz"


def emb_path(reid, seq):
    return ROOT / "cache" / "emb" / DET / reid / f"{seq}.npy"


def gmc_path(seq):
    return ROOT / "cache" / "gmc" / f"{seq}.npy"


def cached_ready(seq, reid):
    ok = det_path(seq).exists() and gmc_path(seq).exists()
    return ok and (reid == NO_REID or emb_path(reid, seq).exists())


@st.cache_data
def load_dets_c(seq):
    return load_dets(det_path(seq))


@st.cache_data(max_entries=8)
def load_emb_c(reid, seq):
    if reid == NO_REID:
        return None
    e = np.load(emb_path(reid, seq)).astype(np.float32)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return np.divide(e, n, out=np.zeros_like(e), where=n > 0)


def run_tags():
    d = RUN_DIR / "trackers"
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "timing.json").exists())


@st.cache_data(ttl=30)
def metrics():
    p = RUN_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(ttl=30)
def speed():
    p = RUN_DIR / "speed.json"
    return json.loads(p.read_text()) if p.exists() else None


def tag_label(tag):
    tr, reid = tag.split("__", 1)
    return f"{TRACKER_NAMES.get(tr, tr)} + {reid}"


def reid_options(include_none=True):
    return REID_KEYS + ([NO_REID] if include_none else [])


# ----------------------------------------------------------------------------------- leaderboard
def page_leaderboard():
    st.title("Leaderboard")
    st.caption(f"{CFG['dataset']['name']} (MOT17 train, 7 sequences, first 50% of frames) · detector "
               f"`{DET}` · TrackEval, pedestrian class · online results (no interpolation)")
    m = metrics()
    if not m:
        st.info("No `metrics.json` yet - the benchmark is still running "
                "(`python scripts/run_benchmark.py`). Pages that use the caches already work.")
        return
    rows = []
    for tag, per in m.items():
        tr, reid = tag.split("__", 1)
        c = per["COMBINED_SEQ"]
        rows.append({"Tracker": TRACKER_NAMES.get(tr, tr), "ReID": reid, **{k: c[k] for k in
                     ["HOTA", "AssA", "IDF1", "IDSW", "DetA", "MOTA", "FP", "FN"]}})
    df = pd.DataFrame(rows)
    sp = speed()
    if sp:
        df["ReID ms/crop"] = df["ReID"].map(lambda r: sp["reid"].get(r, {}).get("ms_per_crop_bs32", np.nan))

    best = df.loc[df["HOTA"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best HOTA", f"{best['HOTA']:.2f}", help=f"{best['Tracker']} + {best['ReID']}")
    c1.caption(f"{best['Tracker']} + `{best['ReID']}`")
    bi = df.loc[df["IDF1"].idxmax()]
    c2.metric("Best IDF1", f"{bi['IDF1']:.2f}")
    c2.caption(f"{bi['Tracker']} + `{bi['ReID']}`")
    bs = df.loc[df["IDSW"].idxmin()]
    c3.metric("Fewest ID switches", int(bs["IDSW"]))
    c3.caption(f"{bs['Tracker']} + `{bs['ReID']}`")
    c4.metric("Runs evaluated", len(df))
    c4.caption(f"{df['Tracker'].nunique()} trackers × {df['ReID'].nunique()} ReID settings")

    import altair as alt

    metric = st.segmented_control("Metric", ["HOTA", "AssA", "IDF1", "IDSW", "MOTA"], default="HOTA")
    order = [k for k in reid_options() if k in set(df["ReID"])]
    chart = (alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=18)
             .encode(x=alt.X("ReID:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
                     y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=metric != "HOTA")),
                     color=alt.Color("ReID:N", scale=alt.Scale(domain=order, range=[COLOR_OF[k] for k in order]),
                                     legend=alt.Legend(orient="top", title=None)),
                     column=alt.Column("Tracker:N", title=None, sort=list(TRACKER_NAMES.values()),
                                       header=alt.Header(labelFontSize=13, labelOrient="bottom")),
                     tooltip=["Tracker", "ReID", alt.Tooltip(f"{metric}:Q", format=".2f"),
                              alt.Tooltip("IDF1:Q", format=".2f"), "IDSW"])
             .properties(width=170, height=280)
             .configure_view(stroke=None).configure_axis(gridColor="#8882", domain=False))
    st.altair_chart(chart)

    st.subheader("All runs")
    st.dataframe(df.sort_values(["Tracker", "HOTA"], ascending=[True, False]), hide_index=True,
                 column_config={k: st.column_config.NumberColumn(format="%.2f") for k in
                                ["HOTA", "AssA", "IDF1", "DetA", "MOTA", "ReID ms/crop"]})

    st.subheader("Per sequence")
    seq_rows = []
    for tag, per in m.items():
        tr, reid = tag.split("__", 1)
        for s, v in per.items():
            if s != "COMBINED_SEQ":
                seq_rows.append({"run": f"{TRACKER_NAMES.get(tr, tr)} + {reid}", "sequence": s, "HOTA": v["HOTA"],
                                 "IDF1": v["IDF1"], "IDSW": v["IDSW"]})
    sdf = pd.DataFrame(seq_rows)
    pm = st.segmented_control("Per-sequence metric", ["HOTA", "IDF1", "IDSW"], default="HOTA", key="psm")
    piv = sdf.pivot(index="run", columns="sequence", values=pm)
    st.dataframe(piv.style.format("{:.1f}" if pm != "IDSW" else "{:.0f}")
                 .background_gradient(cmap="Blues" if pm != "IDSW" else "Oranges", axis=0))
    if sp:
        st.subheader("Speed (isolated, CPU)")
        st.caption(f"{sp['threads']} threads · detector {sp['detector_s_per_frame']:.2f} s/frame")
        st.dataframe(pd.DataFrame(sp["reid"]).T.rename(columns=lambda c: c.replace("_", " ")),
                     column_config={c: st.column_config.NumberColumn(format="%.1f")
                                    for c in pd.DataFrame(sp["reid"]).T.rename(columns=lambda c: c.replace("_", " ")).columns})


# ---------------------------------------------------------------------------------- video viewer
def frame_image(seq_name, f, tracks_by_f, sw, gt_by_f, show_gt, title, trail):
    s = sequences()[seq_name]
    img = s.image(f)
    trails = None
    if trail:
        trails = {}
        for ff in range(max(1, f - trail), f + 1):
            for r in tracks_by_f.get(ff, []):
                trails.setdefault(int(r[1]), []).append((r[2] + r[4] / 2, r[3] + r[5]))
        alive = {int(r[1]) for r in tracks_by_f.get(f, [])}
        trails = {k: v for k, v in trails.items() if k in alive}
    im = draw_frame(img, tracks_by_f.get(f), gt_by_f.get(f) if show_gt else None, switched=sw.get(f),
                    title=title, trails=trails)
    return im[:, :, ::-1]


@st.cache_data(max_entries=16)
def run_tracks(tag, seq):
    return load_mot(RUN_DIR / "trackers" / tag / "data" / f"{seq}.txt")


def page_viewer():
    st.title("Video viewer")
    tags = [t for t in run_tags() if (RUN_DIR / "trackers" / t / "data").exists()]
    if not tags:
        st.info("No tracking runs on disk yet.")
        return
    c = st.columns([1, 2, 2])
    seq = c[0].selectbox("Sequence", list(sequences()))
    a = c[1].selectbox("Run A", tags, format_func=tag_label, index=0)
    b = c[2].selectbox("Run B", tags, format_func=tag_label, index=min(1, len(tags) - 1))
    o = st.columns(4)
    show_gt = o[0].toggle("Show GT boxes", False)
    trail = o[1].slider("Trail (frames)", 0, 60, 20)
    mode = o[2].radio("Mode", ["Scrub frames", "Render MP4"], horizontal=True)
    s = sequences()[seq]
    gt = gt_of(seq)
    gt_by = by_frame(gt)
    runs = {}
    for t in (a, b):
        tr = run_tracks(t, seq)
        runs[t] = (by_frame(tr), id_switch_frames(tr, gt), len(np.unique(tr[:, 1])) if len(tr) else 0)
    st.caption("Red boxes = display-time ID switch vs GT (the GT identity under the box was last covered by "
               "another track id). Official IDSW counts come from TrackEval.")
    if mode == "Scrub frames":
        f = st.slider("Frame", 1, s.length, 1)
        cols = st.columns(2)
        for col, t in zip(cols, (a, b)):
            tb, sw, nid = runs[t]
            col.image(frame_image(seq, f, tb, sw, gt_by, show_gt, tag_label(t), trail), use_container_width=True)
            n_sw = sum(len(v) for k, v in sw.items() if k <= f)
            col.caption(f"{nid} track ids in sequence · {n_sw} display ID switches up to frame {f}")
    else:
        cols = st.columns(2)
        for col, t in zip(cols, (a, b)):
            out = RUN_DIR / "videos" / t / f"{seq}.mp4"
            if not out.exists():
                with col:
                    bar = st.progress(0.0, text=f"Rendering {tag_label(t)} ...")
                    render_sequence_video(s, out, tracks=run_tracks(t, seq), gt=gt, title=tag_label(t), scale=0.5,
                                          show_gt=show_gt, trail_len=trail, progress=lambda p: bar.progress(p))
                    bar.empty()
            col.video(str(out))
            col.caption(tag_label(t))


# ----------------------------------------------------------------------------------- live replay
def page_live():
    st.title("Live replay")
    st.caption("A benchmark sequence is streamed as a simulated camera feed and tracked online, frame by "
               "frame, from the cached detections / embeddings / camera motion (identical to the benchmark run). "
               "For the full live chain including the detector, use **Test videos**.")
    c = st.columns(4)
    seq = c[0].selectbox("Sequence", list(sequences()))
    tracker = c[1].selectbox("Tracker", list(TRACKER_NAMES), format_func=TRACKER_NAMES.get)
    reids = [r for r in reid_options(tracker != "tracktrack") if cached_ready(seq, r)]
    if not reids:
        st.warning("Caches for this sequence are not ready yet.")
        return
    reid = c[2].selectbox("ReID", reids)
    speedup = c[3].select_slider("Playback", [0.25, 0.5, 1.0, 2.0, 4.0, "max"], value=1.0)
    o = st.columns(4)
    show_dets = o[0].toggle("Show detections", False)
    show_gt = o[1].toggle("Show GT", False)
    trail = o[2].slider("Trail", 0, 60, 30)
    start_f = o[3].number_input("Start frame", 1, sequences()[seq].length, 1)
    go = st.button("▶ Start stream", type="primary")
    view = st.empty()
    stats = st.empty()
    if not go:
        view.image(sequences()[seq].image(int(start_f))[:, :, ::-1], use_container_width=True,
                   caption="press Start - the stream restarts on every change")
        return
    from reidbench.online import CachedSource, OnlineTracker

    s = sequences()[seq]
    src = CachedSource(load_dets_c(seq), load_emb_c(reid, seq), np.load(gmc_path(seq)))
    trk = OnlineTracker(tracker, CFG["trackers"][tracker], s, src, reid != NO_REID, CFG["output_filter"])
    gt_by = by_frame(gt_of(seq))
    hist, ids_seen, t_trk = {}, set(), []
    period = None if speedup == "max" else 1.0 / (s.fps * float(speedup))
    t_next = time.perf_counter()
    for f in range(1, s.length + 1):
        rows, std, t = trk.step(f)
        t_trk.append(t["tracker_ms"])
        if f < start_f:
            continue
        for r in rows:
            hist.setdefault(int(r[1]), []).append((r[2] + r[4] / 2, r[3] + r[5]))
            hist[int(r[1])] = hist[int(r[1])][-max(trail, 1):]
            ids_seen.add(int(r[1]))
        alive = {int(r[1]) for r in rows}
        im = draw_frame(s.image(f), rows, gt_by.get(f) if show_gt else None,
                        dets=std if show_dets else None,
                        trails={k: v for k, v in hist.items() if k in alive} if trail else None,
                        title=f"{seq} #{f}  {TRACKER_NAMES[tracker]} + {reid}")
        if period:
            t_next += period
            time.sleep(max(0.0, t_next - time.perf_counter()))
        view.image(im[:, :, ::-1], use_container_width=True)
        stats.markdown(f"**frame** {f}/{s.length} · **active tracks** {len(rows)} · **ids so far** "
                       f"{len(ids_seen)} · **detections** {len(std)} · **association** "
                       f"{np.mean(t_trk[-30:]):.1f} ms/frame")
    st.success("Stream finished.")


# ------------------------------------------------------------------------------------ test bench
PARAM_SPECS = {  # (min, max, step) for the sliders; anything else in the config is shown read-only
    "deepocsort": {"det_thresh": (0.1, 0.9, 0.05), "iou_threshold": (0.05, 0.6, 0.05),
                   "w_association_emb": (0.0, 1.5, 0.05), "alpha_fixed_emb": (0.5, 1.0, 0.01),
                   "aw_param": (0.0, 1.0, 0.05), "inertia": (0.0, 0.5, 0.05), "max_age": (5, 120, 5),
                   "min_hits": (1, 6, 1)},
    "tracktrack": {"det_thr": (0.1, 0.9, 0.05), "init_thr": (0.1, 0.95, 0.05), "match_thr": (0.3, 0.95, 0.05),
                   "penalty_p": (0.0, 0.5, 0.05), "penalty_q": (0.0, 0.8, 0.05), "tai_thr": (0.2, 0.9, 0.05),
                   "max_time_lost_s": (0.5, 5.0, 0.5)},
    "fasttracker": {"track_thresh": (0.1, 0.9, 0.05), "match_thresh": (0.3, 0.95, 0.05),
                    "track_buffer": (5, 120, 5), "proximity_thresh": (0.1, 0.9, 0.05),
                    "appearance_thresh": (0.05, 0.6, 0.05), "init_iou_suppress": (0.3, 0.95, 0.05)},
}


def evaluate_rows(results: dict, seq_names: list) -> dict:
    """results: {seq: [mot lines]} -> TrackEval summary per seq + combined."""
    from reidbench.evaluate import evaluate

    work = Path(tempfile.mkdtemp(prefix="bench_", dir=ROOT / "cache"))
    try:
        seqs = [sequences()[s] for s in seq_names]
        seqmap = write_eval_gt(seqs, work / "gt")
        d = work / "trk" / "bench" / "data"
        d.mkdir(parents=True)
        for s, rows in results.items():
            (d / f"{s}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        return evaluate(work / "gt", seqmap, work / "trk", ["bench"], threads=1)["bench"]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def page_bench():
    from reidbench.runners import run_sequence

    st.title("Test bench")
    st.caption("Re-run any tracker × ReID combination with your own hyper-parameters on the cached "
               "detections/embeddings and score it against GT. Association-only, so a full sequence takes seconds.")
    c = st.columns([1, 1, 2])
    tracker = c[0].selectbox("Tracker", list(TRACKER_NAMES), format_func=TRACKER_NAMES.get, key="bt")
    opts = reid_options(tracker != "tracktrack")
    ready = [r for r in opts if any(cached_ready(s, r) for s in sequences())]
    if not ready:
        st.warning("No ReID setting has ready caches for this tracker yet - the benchmark is still running.")
        return
    reid = c[1].selectbox("ReID", ready, key="br",
                          help="only settings whose detections/embeddings are cached are listed")
    avail = [s for s in sequences() if cached_ready(s, reid)]
    seq_sel = c[2].multiselect("Sequences", avail, default=avail[:2])
    base = CFG["trackers"][tracker]
    with st.expander("Hyper-parameters", expanded=True):
        cols = st.columns(4)
        params = dict(base)
        for i, (k, (lo, hi, step)) in enumerate(PARAM_SPECS[tracker].items()):
            v = base[k]
            if isinstance(step, int) and isinstance(v, int):
                params[k] = cols[i % 4].slider(k, lo, hi, v, step, key=f"{tracker}_{k}")
            else:
                params[k] = cols[i % 4].slider(k, float(lo), float(hi), float(v), float(step), key=f"{tracker}_{k}")
        changed = {k: v for k, v in params.items() if v != base[k]}
        st.caption("Changed from benchmark default: " + (", ".join(f"`{k}`={v}" for k, v in changed.items())
                                                         or "nothing"))
    if not seq_sel:
        st.warning("Pick at least one sequence with ready caches.")
        return
    if not st.button("Run & evaluate", type="primary"):
        return

    def run(p):
        res, secs, frames = {}, 0.0, 0
        for sname in seq_sel:
            s = sequences()[sname]
            rows, t = run_sequence(tracker, p, s, load_dets_c(sname), load_emb_c(reid, sname),
                                   np.load(gmc_path(sname)), CFG["output_filter"])
            res[sname], secs, frames = rows, secs + t, frames + s.length
        return res, frames / max(secs, 1e-9)

    with st.spinner("Tracking + TrackEval ..."):
        res_new, fps_new = run(params)
        m_new = evaluate_rows(res_new, seq_sel)
        if changed:
            res_def, _ = run(base)
            m_def = evaluate_rows(res_def, seq_sel)
        else:
            m_def = m_new
    keys = ["HOTA", "AssA", "IDF1", "IDSW", "DetA", "MOTA"]
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys):
        n, d = m_new["COMBINED_SEQ"][k], m_def["COMBINED_SEQ"][k]
        delta = n - d
        col.metric(k, f"{n:.2f}" if k != "IDSW" else int(n),
                   delta=(f"{delta:+.2f}" if k != "IDSW" else f"{int(delta):+d}") if changed else None,
                   delta_color="inverse" if k == "IDSW" else "normal")
    st.caption(f"Association speed {fps_new:.0f} FPS · deltas vs the benchmark default parameters on the same sequences")
    df = pd.DataFrame({s: {k: v[k] for k in keys} for s, v in m_new.items()}).T
    st.dataframe(df, column_config={k: st.column_config.NumberColumn(format="%.2f") for k in keys if k != "IDSW"})
    st.session_state["bench_last"] = {"tracker": tracker, "reid": reid, "params": params, "metrics": m_new}
    st.download_button("Download parameters (YAML)", yaml.safe_dump({"trackers": {tracker: params}}),
                       file_name=f"{tracker}_params.yaml")


# ----------------------------------------------------------------------------------- test videos
@st.cache_resource
def get_detector(name):
    from ultralytics import YOLO

    return YOLO(str(ROOT / "weights" / f"{name}.pt"))


@st.cache_resource
def get_embedder(reid):
    from reidbench.reid.backends import build_embedder

    return build_embedder(CFG["reid_models"][reid], threads=CFG.get("threads"), name=reid)


def page_videos():
    from reidbench.online import LiveSource, OnlineTracker, VideoSequence
    from reidbench.viz import H264Writer

    st.title("Test videos")
    st.caption("Full live chain on a video: detector → ReID → camera-motion → tracker, every frame. "
               "Use a simulated test video (MOT17 sequences encoded as camera feeds, `scripts/make_videos.py --raw`) "
               "or upload your own clip. CPU-only: the detector dominates the cost.")
    vids = sorted((ROOT / "data" / "videos").glob("*.mp4"))
    src = st.radio("Source", ["Simulated test video", "Upload"], horizontal=True)
    path = None
    if src == "Simulated test video":
        if not vids:
            st.info("No simulated test videos yet - run `python scripts/make_videos.py --raw`.")
        else:
            path = st.selectbox("Video", vids, format_func=lambda p: p.name)
    else:
        up = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm"])
        if up is not None:
            h = hashlib.sha1(up.getvalue()).hexdigest()[:12]
            path = ROOT / "cache" / "uploads" / f"{h}_{Path(up.name).name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(up.getvalue())
    c = st.columns(5)
    tracker = c[0].selectbox("Tracker", list(TRACKER_NAMES), format_func=TRACKER_NAMES.get, key="vt")
    reid = c[1].selectbox("ReID", reid_options(tracker != "tracktrack"), key="vr")
    det_name = c[2].selectbox("Detector", ["yolo26n", "yolo26l"],
                              help="yolo26n is ~10x faster on CPU; the benchmark uses yolo26l @1280")
    imgsz = c[3].select_slider("Detector size", [640, 960, 1280], value=960)
    max_frames = c[4].number_input("Max frames", 10, 3000, 150, 10)
    if path is None or not st.button("Run tracking", type="primary"):
        return
    seq = VideoSequence.from_file(path, max_frames=int(max_frames))
    live = LiveSource(get_detector(det_name),
                      dict(imgsz=imgsz, conf=CFG["detector"]["conf"], iou=CFG["detector"]["nms_loose"], classes=[0],
                           max_det=1000, agnostic_nms=True),
                      get_embedder(reid) if reid != NO_REID else None,
                      std_iou=CFG["detector"]["nms_std"])
    trk = OnlineTracker(tracker, CFG["trackers"][tracker], seq, live, reid != NO_REID, CFG["output_filter"])
    out = ROOT / "cache" / "video_runs" / f"{path.stem}__{tracker}__{reid}__{det_name}{imgsz}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = seq.width // 2 * 2, seq.height // 2 * 2
    wr = H264Writer(out, seq.fps, W, H)
    view, bar, stat = st.empty(), st.progress(0.0), st.empty()
    timings, hist, ids, all_rows = [], {}, set(), []
    try:
        for f in range(1, seq.length + 1):
            img = seq.image(f)
            rows, std, t = trk.step(f, img)
            timings.append(t)
            all_rows.append(rows)
            for r in rows:
                hist.setdefault(int(r[1]), []).append((r[2] + r[4] / 2, r[3] + r[5]))
                hist[int(r[1])] = hist[int(r[1])][-30:]
                ids.add(int(r[1]))
            alive = {int(r[1]) for r in rows}
            im = draw_frame(img, rows, trails={k: v for k, v in hist.items() if k in alive},
                            title=f"{seq.name} #{f}  {TRACKER_NAMES[tracker]} + {reid}")
            wr.write(cv2_resize(im, W, H))
            if f % 3 == 1 or f == seq.length:
                view.image(im[:, :, ::-1], use_container_width=True)
            bar.progress(f / seq.length, text=f"frame {f}/{seq.length}")
            tm = pd.DataFrame(timings[-20:]).mean()
            stat.markdown(" · ".join(f"**{k.replace('_ms', '')}** {v:.0f} ms" for k, v in tm.items()) +
                          f" · **tracks** {len(rows)} · **ids** {len(ids)}")
    finally:
        wr.close()
    bar.empty()
    tm = pd.DataFrame(timings).mean()
    st.success(f"Done: {seq.length} frames, {len(ids)} identities, "
               f"{1000 / tm.sum():.2f} FPS end-to-end on this CPU.")
    st.video(str(out))
    rows = np.concatenate(all_rows) if all_rows else np.zeros((0, 7))
    mot = "\n".join(f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.3f},-1,-1,-1"
                    for r in rows)
    d1, d2 = st.columns(2)
    d1.download_button("Download annotated MP4", out.read_bytes(), file_name=out.name, mime="video/mp4")
    d2.download_button("Download tracks (MOT txt)", mot, file_name=f"{out.stem}.txt")


def cv2_resize(im, W, H):
    import cv2

    return im if (im.shape[1], im.shape[0]) == (W, H) else cv2.resize(im, (W, H))


# ------------------------------------------------------------------------------------------- main
PAGES = {"Leaderboard": page_leaderboard, "Video viewer": page_viewer, "Live replay": page_live,
         "Test bench": page_bench, "Test videos": page_videos}

with st.sidebar:
    st.header("ReID Benchmark Lab")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption(f"Dataset `{CFG['dataset']['name']}` · detector `{DET}`")
    n_ready = {r: sum(cached_ready(s, r) for s in sequences()) for r in REID_KEYS}
    st.caption("Embedding caches: " + ", ".join(f"`{k}` {v}/{len(sequences())}" for k, v in n_ready.items()))
    st.caption(f"Evaluated runs: {len(metrics())}")

PAGES[page]()
