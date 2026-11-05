"""MOT-format dataset helpers."""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class Sequence:
    name: str
    root: Path
    frames: list  # sorted image paths, frame i (1-based) == frames[i-1]
    fps: float
    width: int
    height: int

    @property
    def length(self) -> int:
        return len(self.frames)

    def image(self, frame_id: int):
        return cv2.imread(str(self.frames[frame_id - 1]))


def load_sequences(split_dir: str | Path, names: list[str] | None = None) -> list[Sequence]:
    split_dir = Path(split_dir)
    seqs = []
    for d in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if names and d.name not in names:
            continue
        ini = configparser.ConfigParser()
        ini.read(d / "seqinfo.ini")
        info = ini["Sequence"]
        img_dir = d / info.get("imDir", "img1")
        frames = sorted(img_dir.glob("*" + info.get("imExt", ".jpg")))
        seqs.append(Sequence(d.name, d, frames, float(info.get("frameRate", 30)),
                             int(info["imWidth"]), int(info["imHeight"])))
    return seqs


def write_eval_gt(seqs: list[Sequence], out_dir: str | Path) -> Path:
    """Create a TrackEval-ready GT folder (gt.txt + seqinfo.ini with the *actual* number of frames,
    some MOT17 subsets ship the original seqLength) and a seqmap file."""
    out_dir = Path(out_dir)
    for s in seqs:
        d = out_dir / s.name
        (d / "gt").mkdir(parents=True, exist_ok=True)
        lines = (s.root / "gt" / "gt.txt").read_text().splitlines()
        keep = [l for l in lines if l and int(float(l.split(",")[0])) <= s.length]
        (d / "gt" / "gt.txt").write_text("\n".join(keep) + "\n")
        (d / "seqinfo.ini").write_text(
            "[Sequence]\nname={n}\nimDir=img1\nframeRate={f:g}\nseqLength={L}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n"
            .format(n=s.name, f=s.fps, L=s.length, w=s.width, h=s.height))
    seqmap = out_dir / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(s.name for s in seqs) + "\n")
    return seqmap
