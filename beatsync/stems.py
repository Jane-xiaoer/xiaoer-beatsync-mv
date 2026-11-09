#!/usr/bin/env python3
"""小耳音乐卡点MV · 分轨 —— Demucs 把歌拆成 4 轨（鼓 / 贝斯 / 人声 / 其他）或 6 轨（再拆出吉他、钢琴）。

  beatsync stems song.mp3 [-o 输出目录] [--six]

为什么要分轨：只听整轨，鼓、贝斯、人声、旋律搅在一起，只能卡到「拍」；分开后每一轨的每个起音
都能变成一个看得见的动作 —— 鼓推镜头、贝斯管光的呼吸、人声起句切镜头、旋律推画面往前走。
（小耳的经验：四轨就有明显感觉；六轨适合钢琴、吉他是主角的歌。）

时间基准：所有时间以原曲无缝解码为准。Demucs 读 mp3 时可能带编码器延迟（pdoom 实测 23ms），
这里把四轨相加和原曲做互相关，实测偏移并写进 stems.json，读分轨时自动补回。
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

SR = 44100
FOUR = ["drums", "bass", "vocals", "other"]
SIX = FOUR + ["guitar", "piano"]


def measure_offset(audio, stem_paths, max_lag=4096, seconds=60.0):
    """分轨相加 vs 原曲的样本偏移（正数 = 分轨晚）。改写自 pdoom-video common.stem_offset_samples（MIT）。"""
    import librosa
    import soundfile as sf
    from scipy.signal import correlate
    n = int(seconds * SR)
    mix, _ = librosa.load(str(audio), sr=SR, mono=True, duration=seconds)
    acc = None
    for p in stem_paths:
        y, s = sf.read(p, dtype="float32", always_2d=True, frames=n + max_lag)
        y = y.mean(axis=1)
        acc = y if acc is None else acc[:min(len(acc), len(y))] + y[:min(len(acc), len(y))]
    m = min(len(mix), len(acc) - max_lag)
    c = correlate(acc[:m + max_lag], mix[:m], mode="valid", method="fft")
    c2 = correlate(mix[:m], acc[:m - max_lag], mode="valid", method="fft")
    pos, neg = int(np.argmax(c)), int(np.argmax(c2))
    return pos if c[pos] >= c2[neg] else -neg


def separate(audio, out_dir, six=False, device=None):
    audio = Path(audio).resolve()
    out_dir = Path(out_dir)
    meta_p = out_dir / "stems.json"
    names = SIX if six else FOUR
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("audio") == str(audio) and meta.get("six") == six and all((out_dir / f"{n}.wav").exists() for n in names):
            return meta
    out_dir.mkdir(parents=True, exist_ok=True)
    model = "htdemucs_6s" if six else "htdemucs"
    if device is None:
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            device = "cpu"
    print(f"  分轨中（{model}，{device}）…首次运行会下载模型", flush=True)
    import torch
    import librosa
    import soundfile as sf
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    m = get_model(model)
    m.eval()
    wav, _ = librosa.load(str(audio), sr=m.samplerate, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    x = torch.from_numpy(wav[:m.audio_channels]).float()
    ref = x.mean(0)
    x = (x - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(m, x[None], device=device, shifts=1, split=True, overlap=0.25, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    for i, n in enumerate(m.sources):
        if n in names:
            sf.write(out_dir / f"{n}.wav", out[i].cpu().numpy().T, m.samplerate, subtype="FLOAT")
    off = measure_offset(audio, [out_dir / f"{n}.wav" for n in names])
    meta = {"audio": str(audio), "model": model, "six": six, "stems": names,
            "offset_samples": int(off), "offset_ms": round(off / SR * 1000, 2)}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"  ✓ {len(names)} 轨 · 分轨与原曲偏移 {meta['offset_ms']}ms（已自动补偿）")
    return meta


def load(out_dir, name, n_samples=None):
    """读一轨，按实测偏移对齐到原曲时间轴，单声道 44.1k。"""
    import soundfile as sf
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "stems.json").read_text())
    y, s = sf.read(out_dir / f"{name}.wav", dtype="float32", always_2d=True)
    off = meta["offset_samples"]
    y = y[off:] if off >= 0 else np.pad(y, ((-off, 0), (0, 0)))
    y = y.mean(axis=1)
    if n_samples is not None:
        y = y[:n_samples] if len(y) >= n_samples else np.pad(y, (0, n_samples - len(y)))
    return y


def main(argv=None):
    ap = argparse.ArgumentParser(description="小耳音乐卡点MV · 分轨")
    ap.add_argument("audio")
    ap.add_argument("-o", "--out")
    ap.add_argument("--six", action="store_true", help="6 轨（多拆吉他、钢琴）")
    a = ap.parse_args(argv)
    out = a.out or str(Path(a.audio).with_suffix("")) + ".stems"
    m = separate(a.audio, out, a.six)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
