#!/usr/bin/env python3
"""小耳音乐卡点MV · 第 4 步：卡点体检 —— 拿成片本身量，每一刀到底差几帧。

  beatsync check mv.mp4 --plan plan.json       # 对照方案：每一刀是否落在计划的那一帧
  beatsync check 任意成片.mp4 --cues cues.json  # 不需要方案：量任何剪辑（剪映/PR 剪的也行）离拍有多远

体检只认成片：从 mp4 逐帧解码、算帧差找出真实的切点，不看作者导出的静帧或工程文件。
另外把成片里的音轨和原曲做互相关，量出音画整体偏移（片头多了一帧黑场、音频被重采样拖慢，都会在这里现形）。
输出 HTML 报告：时间轴（能量曲线 + 拍 + 切点）、误差分布、偏得最多的切点前后帧截图。
"""
import argparse
import base64
import html
import json
import subprocess
from pathlib import Path

import numpy as np

W, H = 96, 54


def decode_gray(video, fps):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                          f"fps={fps},scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    n = len(raw) // (W * H)
    return np.frombuffer(raw[:n * W * H], np.uint8).reshape(n, H, W).astype(np.float32)


def detect_cuts(frames):
    """切点 = 前后两帧「不是同一幅画」。每帧减均值、除标准差后算相邻帧相关系数：
    同一镜头里推拉、闪亮、底鼓顶一下，相关性都在 0.85 以上；换了画面会掉到 0.5 以下。
    相关性 < 0.75 且比前后 1 秒的中位数低 0.15 以上 → 切点（返回切后第一帧的帧号）。纯色帧不判。"""
    f = frames.reshape(len(frames), -1)
    sd = f.std(1)
    z = (f - f.mean(1, keepdims=True)) / (sd[:, None] + 1e-6)
    corr = np.mean(z[1:] * z[:-1], axis=1)
    flat = (sd[1:] < 2) | (sd[:-1] < 2)
    corr[flat] = 1.0
    cuts = []
    for i in range(len(corr)):
        lo, hi = max(0, i - 15), min(len(corr), i + 16)
        ref = np.median(corr[lo:hi])
        if corr[i] < 0.75 and corr[i] < ref - 0.15 and corr[i] == corr[max(0, i - 2):i + 3].min():
            cuts.append(i + 1)
    return cuts, 1 - corr


def action_sync(frames, onsets, fps, cuts):
    """起音即动作：军鼓那一帧画面是不是真的亮了一下。看平均亮度的逐帧变化，在军鼓 ±3 帧里找最大上跳，
    记它离军鼓差几帧（切点附近 ±2 帧的跳过，切镜本身会改变亮度）。"""
    lum = frames.reshape(len(frames), -1).mean(1)
    dl = np.diff(lum, prepend=lum[0])
    cut_set = np.array(cuts) if cuts else np.array([-99])
    offs = []
    for t, s, *_ in onsets:
        if s < 0.5:
            continue
        f0 = int(round(t * fps))
        if f0 < 3 or f0 >= len(lum) - 3 or np.min(np.abs(cut_set - f0)) <= 2:
            continue
        w = dl[f0 - 3:f0 + 4]
        if w.max() <= 0.3:
            continue
        offs.append(int(np.argmax(w)) - 3)
    return offs


def audio_env(path, sr=8000, hop=80):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True).stdout
    y = np.frombuffer(raw, np.float32)
    n = len(y) // hop
    e = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1))
    return np.maximum(0, np.diff(e, prepend=e[0])), sr / hop


def av_offset_ms(video, audio):
    """成片音轨 vs 原曲：前 60 秒起音包络互相关，正数 = 成片里的声音比原曲晚。"""
    a, r = audio_env(video)
    b, _ = audio_env(audio)
    n = int(min(len(a), len(b), r * 60))
    if n < r * 2:
        return None
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    maxlag = int(r * 0.5)
    best, lag = -1e18, 0
    for L in range(-maxlag, maxlag + 1):
        x = np.dot(a[max(0, L):n + min(0, L)], b[max(0, -L):n - max(0, L)])
        if x > best:
            best, lag = x, L
    return lag / r * 1000


def thumb_pair(video, frame, fps, out_w=240):
    """切点前一帧 + 切点那一帧，拼成一张小图（base64，直接嵌进报告）。"""
    t0 = max(0, (frame - 1) / fps)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-i", str(video),
                          "-frames:v", "2", "-vf", f"scale={out_w}:-2,tile=2x1", "-f", "image2",
                          "-c:v", "mjpeg", "-q:v", "5", "-"], capture_output=True).stdout
    return base64.b64encode(raw).decode() if raw else ""


def check(video, plan=None, cues=None, audio=None, out=None):
    if plan:
        P = json.loads(Path(plan).read_text())
        cues = cues or P["meta"]["cues"]
        audio = audio or P["meta"]["audio"]
    C = json.loads(Path(cues).read_text())
    fps = (P["meta"]["fps"] if plan else C["meta"]["fps"])
    frames = decode_gray(video, fps)
    found, diff = detect_cuts(frames)
    rep = {"video": str(video), "fps": fps, "frames": len(frames), "found_cuts": len(found)}

    beat_frames = np.array([round(b["t"] * fps) for b in C["beats"]])
    down_frames = np.array([round(b["t"] * fps) for b in C["downbeats"]])
    rows = []
    if plan:
        planned = [s["frame_in"] for s in P["shots"]][1:]       # 第一镜从 0 开始，不算切点
        why = {s["frame_in"]: s["cut_why"] for s in P["shots"]}
        fa = np.array(found) if found else np.array([10 ** 9])
        for pf in planned:
            j = int(np.argmin(np.abs(fa - pf)))
            off = int(fa[j] - pf)
            rows.append({"frame": pf, "t": pf / fps, "found": int(fa[j]) if abs(off) <= 6 else None,
                         "off": off if abs(off) <= 6 else None, "why": why.get(pf, "")})
        offs = [r["off"] for r in rows if r["off"] is not None]
        missing = sum(1 for r in rows if r["off"] is None)
        extra = len([f for f in found if np.min(np.abs(np.array(planned) - f)) > 6]) if planned else len(found)
        rep.update(mode="对照方案", planned=len(planned), missing=missing, extra=extra)
    else:
        for f in found:
            j = int(np.argmin(np.abs(beat_frames - f)))
            rows.append({"frame": f, "t": f / fps, "found": f, "off": int(f - beat_frames[j]),
                         "why": "最近的拍" + ("（强拍）" if beat_frames[j] in down_frames else "")})
        offs = [r["off"] for r in rows]
        rep.update(mode="量离拍距离")
    offs = np.array(offs) if len(offs) else np.array([0])
    rep.update(exact=int(np.sum(offs == 0)), within1=int(np.sum(np.abs(offs) <= 1)),
               within2=int(np.sum(np.abs(offs) <= 2)), measured=len(rows) - rep.get("missing", 0),
               mean_abs_frames=round(float(np.mean(np.abs(offs))), 3),
               max_abs_frames=int(np.max(np.abs(offs))))
    st = C.get("stems")
    if st and st["onsets"].get("snare"):
        ao = action_sync(frames, st["onsets"]["snare"], fps, found)
        if ao:
            ao = np.array(ao)
            rep["action"] = {"what": "军鼓 → 画面亮一下", "n": int(len(ao)), "exact": int(np.sum(ao == 0)),
                             "within1": int(np.sum(np.abs(ao) <= 1))}
    if audio:
        rep["av_offset_ms"] = None if (o := av_offset_ms(video, audio)) is None else round(o, 1)

    out = Path(out or Path(video).with_suffix(".check.html"))
    worst = sorted([r for r in rows if r["off"] is not None], key=lambda r: -abs(r["off"]))[:6]
    for r in worst:
        r["img"] = thumb_pair(video, r["found"], fps)
    out.write_text(report_html(rep, rows, worst, C, diff, fps))
    out.with_suffix(".json").write_text(json.dumps({"summary": rep, "cuts": [{k: v for k, v in r.items() if k != "img"} for r in rows]},
                                                   ensure_ascii=False, indent=1))
    return rep, out


def report_html(rep, rows, worst, C, diff, fps):
    dur = rep["frames"] / fps
    Wd = 1100

    def x(t):
        return f"{t / dur * Wd:.1f}"
    e = C["energy"]["values"]
    pts = " ".join(f"{x(i / C['energy']['rate'])},{70 - v * 60:.1f}" for i, v in enumerate(e) if i / C["energy"]["rate"] <= dur)
    beats = "".join(f'<line x1="{x(b["t"])}" x2="{x(b["t"])}" y1="{"72" if b["beat"] == 1 else "76"}" y2="80" class="beat"/>'
                    for b in C["beats"] if b["t"] <= dur)
    cutl = "".join(
        f'<line x1="{x(r["t"])}" x2="{x(r["t"])}" y1="0" y2="70" class="{"ok" if r["off"] == 0 else ("near" if r["off"] is not None and abs(r["off"]) <= 1 else "bad")}"><title>{r["t"]:.2f}s 偏 {r["off"]} 帧 · {html.escape(r["why"])}</title></line>'
        for r in rows)
    secs = "".join(f'<rect x="{x(s["start"])}" width="{float(x(s["end"])) - float(x(s["start"])):.1f}" y="0" height="80" class="sec {s["level"]}"><title>{s["level"]} {s["start"]:.1f}–{s["end"]:.1f}s</title></rect>'
                   for s in C["sections"])
    from collections import Counter
    cnt = Counter(r["off"] for r in rows if r["off"] is not None)
    bars = "".join(f'<div class="hb"><span>{k:+d} 帧</span><i style="width:{cnt[k] / max(cnt.values()) * 100:.0f}%"></i><b>{cnt[k]}</b></div>'
                   for k in sorted(cnt))
    worst_html = "".join(f'<figure><img src="data:image/jpeg;base64,{r["img"]}"><figcaption>{r["t"]:.2f}s · 偏 {r["off"]:+d} 帧<br>{html.escape(r["why"])}</figcaption></figure>'
                         for r in worst if r.get("img"))
    m = rep.get("measured") or 1
    a_ = rep.get("action")
    act_html = (f'<div>起音即动作<b>{a_["exact"] / a_["n"] * 100:.0f}%</b><span class="mut">{a_["what"]}：{a_["exact"]}/{a_["n"]} 同一帧，'
                f'≤1 帧 {a_["within1"] / a_["n"] * 100:.0f}%</span></div>') if a_ else ""
    av = rep.get("av_offset_ms")
    av_txt = "—" if av is None else f"{av:+.0f} ms"
    head = (f"对照方案：计划 {rep['planned']} 刀，找到 {rep['planned'] - rep['missing']} 刀，漏 {rep['missing']}，多出 {rep['extra']}"
            if rep["mode"] == "对照方案" else f"不看方案：成片里找到 {rep['found_cuts']} 刀，逐刀量离最近的拍多远")
    return f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>卡点体检</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#f6f3ec;--fg:#1d1b18;--mut:#7a746a;--line:#d9d2c3;--ok:#1f7a4d;--near:#c98a12;--bad:#c0392b;--acc:#1c3f94}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15140f;--fg:#ece6d8;--mut:#9a927f;--line:#3a362c;--ok:#4cc38a;--near:#e3b341;--bad:#ff6b5b;--acc:#7da2ff}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"PingFang SC",sans-serif;margin:0;padding:32px 16px}}
main{{max-width:1140px;margin:auto}} h1{{font-size:26px;margin:0 0 4px}} .mut{{color:var(--mut)}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}
.kpi div{{border:1px solid var(--line);border-radius:10px;padding:14px}} .kpi b{{display:block;font-size:28px}}
svg{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px}}
.beat{{stroke:var(--mut);stroke-width:.6}} .ok{{stroke:var(--ok);stroke-width:1.2}} .near{{stroke:var(--near);stroke-width:1.6}} .bad{{stroke:var(--bad);stroke-width:2}}
.sec{{fill:transparent}} .sec.最高{{fill:var(--acc);opacity:.10}} .sec.高{{fill:var(--acc);opacity:.05}}
polyline{{fill:none;stroke:var(--acc);stroke-width:1.4}}
.hb{{display:grid;grid-template-columns:70px 1fr 40px;gap:8px;align-items:center;margin:4px 0}} .hb i{{display:block;height:12px;background:var(--acc);border-radius:3px}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}} figure{{margin:0;max-width:260px}} figure img{{width:100%;border-radius:6px}} figcaption{{font-size:12px;color:var(--mut)}}
</style><main>
<h1>卡点体检</h1><div class="mut">{html.escape(Path(rep['video']).name)} · {rep['fps']}fps · {rep['frames']} 帧 · {head}</div>
<div class="kpi">
<div>正好落在那一帧<b>{rep['exact'] / m * 100:.0f}%</b><span class="mut">{rep['exact']}/{m}</span></div>
<div>误差 ≤ 1 帧<b>{rep['within1'] / m * 100:.0f}%</b><span class="mut">{rep['within1']}/{m}</span></div>
<div>平均误差<b>{rep['mean_abs_frames']} 帧</b><span class="mut">最大 {rep['max_abs_frames']} 帧</span></div>
<div>音画整体偏移<b>{av_txt}</b><span class="mut">成片音轨 vs 原曲</span></div>
{act_html}
</div>
<h3>时间轴</h3><div class="mut">蓝线 = 能量；底部刻度 = 拍（长的是强拍）；竖线 = 切点（绿=正好，黄=差 1 帧，红=差更多）；底色深 = 高潮段</div>
<svg viewBox="0 0 {Wd} 80" preserveAspectRatio="none">{secs}<polyline points="{pts}"/>{beats}{cutl}</svg>
<h3>误差分布</h3>{bars}
<h3>偏得最多的切点（切前一帧 | 切后一帧）</h3><div class="grid">{worst_html or '<span class="mut">没有</span>'}</div>
<p class="mut">量法：成片按 {rep['fps']}fps 逐帧解码，缩到 {W}×{H} 灰度，相邻帧归一化相关系数找切点（闪光、推拉不算切）；音画偏移用起音包络互相关（前 60 秒）。</p>
</main></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="小耳音乐卡点MV · 卡点体检")
    ap.add_argument("video")
    ap.add_argument("--plan")
    ap.add_argument("--cues")
    ap.add_argument("--audio", help="原曲（量音画偏移用）")
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    if not (a.plan or a.cues):
        raise SystemExit("✗ 需要 --plan 或 --cues")
    rep, out = check(a.video, a.plan, a.cues, a.audio, a.out)
    m = rep.get("measured") or 1
    print(f"卡点体检 · {rep['mode']}")
    if rep["mode"] == "对照方案":
        print(f"  计划 {rep['planned']} 刀 · 漏 {rep['missing']} · 多出 {rep['extra']}")
    print(f"  正好落在那一帧 {rep['exact']}/{m}（{rep['exact'] / m * 100:.0f}%） · ≤1 帧 {rep['within1'] / m * 100:.0f}% · "
          f"平均 {rep['mean_abs_frames']} 帧 · 最大 {rep['max_abs_frames']} 帧")
    if rep.get("action"):
        a_ = rep["action"]
        print(f"  起音即动作（{a_['what']}）：同一帧 {a_['exact']}/{a_['n']}（{a_['exact'] / a_['n'] * 100:.0f}%） · ≤1 帧 {a_['within1'] / a_['n'] * 100:.0f}%")
    if rep.get("av_offset_ms") is not None:
        print(f"  音画整体偏移 {rep['av_offset_ms']:+.0f} ms")
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
