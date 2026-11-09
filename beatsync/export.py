#!/usr/bin/env python3
"""小耳音乐卡点MV · 第 3 步（B）：导出到剪辑软件 —— 不替你剪成片，把卡好的时间线交回你自己的软件。

  beatsync export plan.json [--format xml|fcpxml|csv|all] [-o 输出前缀] [--with-speed] [--markers structure|bars|beats]

  xml     Final Cut Pro 7 XML（xmeml v5）→ Premiere Pro「文件 › 导入」、DaVinci Resolve「导入时间线」
  fcpxml  FCPXML 1.10 → Final Cut Pro X、DaVinci Resolve
  csv     标记点清单（时间码 / 帧 / 类型 / 说明）→ 任何软件手动对照，或剪映里照着打点

时间线内容：
  V1  每个镜头一个片段，入出点精确到帧（默认 100% 速度；--with-speed 时写入变速）
  A1  原曲，从 0 开始
  标记 段落（带能量档）、乐句起点；--markers bars 再加每个强拍，beats 加每一拍
镜头的推拉运动不写进工程（各软件关键帧格式不通用），写在每个片段的备注里，剪辑师照着做或忽略。
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def dur_frames(path, fps):
    if Path(path).suffix.lower() in IMAGE_EXT:
        return int(fps * 3600)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return int(float(out or 0) * fps)


def tc(frame, fps):
    f = int(fps)
    s, fr = divmod(frame, f)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{fr:02d}"


def markers(plan, cues, level):
    fps = plan["meta"]["fps"]
    mk = []
    for s in cues["sections"]:
        mk.append({"frame": round(s["start"] * fps), "kind": "段落", "name": f"段落 · {s['level']}",
                   "note": f"能量 {s['energy']:.2f}，{s['start']:.2f}–{s['end']:.2f}s"})
    for p in cues["phrases"]:
        mk.append({"frame": p["frame"], "kind": "乐句", "name": f"乐句 · 第{p['bar']}小节", "note": f"置信度 {p['conf']}"})
    if level in ("bars", "beats"):
        for b in cues["downbeats"]:
            mk.append({"frame": b["frame"], "kind": "强拍", "name": f"第{b['bar']}小节", "note": f"置信度 {b['conf']}"})
    if level == "beats":
        for b in cues["beats"]:
            if b["beat"] != 1:
                mk.append({"frame": b["frame"], "kind": "拍", "name": f"{b['bar']}.{b['beat']}", "note": f"置信度 {b['conf']}"})
    for h in cues["hits"]:
        mk.append({"frame": h["frame"], "kind": "重音", "name": "重音", "note": f"强度 {h['conf']}"})
    for b in cues["breaths"]:
        mk.append({"frame": round(b["start"] * fps), "kind": "气口", "name": "气口",
                   "note": f"{b['start']:.2f}–{b['end']:.2f}s 音乐停住"})
    # 同一帧只留一个（优先级：段落 > 乐句 > 重音 > 气口 > 强拍 > 拍）
    pr = {"段落": 0, "乐句": 1, "重音": 2, "气口": 3, "强拍": 4, "拍": 5}
    best = {}
    for m in mk:
        if m["frame"] not in best or pr[m["kind"]] < pr[best[m["frame"]]["kind"]]:
            best[m["frame"]] = m
    return [best[k] for k in sorted(best)]


def shot_note(s):
    return f"{s['cut_why']}；{s['move_why']}（{s['move']['from']}→{s['move']['to']}）；建议速度 ×{s['speed']}"


def src_range(s, fps, with_speed):
    a = int(round(s["src_in"] * fps))
    n = s["frames"]
    return a, a + (int(round(n * s["speed"])) if with_speed else n)


def to_xmeml(plan, cues, mk, with_speed):
    m = plan["meta"]
    fps = int(round(m["fps"]))
    W, H = m["size"].split("x")
    total = plan["shots"][-1]["frame_out"]
    rate = f"<rate><timebase>{fps}</timebase><ntsc>FALSE</ntsc></rate>"
    files, items = {}, []
    for s in plan["shots"]:
        p = s["clip"]
        if p not in files:
            files[p] = {"id": f"file-{len(files) + 1}", "dur": dur_frames(p, fps), "first": True}
        f = files[p]
        a, b = src_range(s, fps, with_speed)
        file_xml = (f'<file id="{f["id"]}"/>' if not f["first"] else
                    f'<file id="{f["id"]}"><name>{escape(Path(p).name)}</name>'
                    f'<pathurl>file://localhost{quote(str(Path(p).resolve()))}</pathurl>{rate}'
                    f'<duration>{f["dur"]}</duration><media><video><samplecharacteristics><width>{W}</width>'
                    f'<height>{H}</height></samplecharacteristics></video></media></file>')
        f["first"] = False
        speed_xml = ""
        if with_speed and abs(s["speed"] - 1) > 1e-3:
            speed_xml = ('<filter><effect><name>Time Remap</name><effectid>timeremap</effectid>'
                         '<effectcategory>motion</effectcategory><effecttype>motion</effecttype><mediatype>video</mediatype>'
                         f'<parameter><parameterid>speed</parameterid><name>speed</name><value>{s["speed"] * 100:.1f}</value></parameter>'
                         '<parameter><parameterid>variablespeed</parameterid><name>variablespeed</name><value>0</value></parameter>'
                         '</effect></filter>')
        items.append(f'<clipitem id="clip-{s["i"]}"><name>{s["i"]:03d} {escape(Path(p).stem)}</name>'
                     f'<duration>{f["dur"]}</duration>{rate}<start>{s["frame_in"]}</start><end>{s["frame_out"]}</end>'
                     f'<in>{a}</in><out>{b}</out>{file_xml}{speed_xml}'
                     f'<comments><mastercomment1>{escape(shot_note(s))}</mastercomment1></comments></clipitem>')
    adur = int(round(cues["meta"]["duration"] * fps))
    audio = (f'<clipitem id="music"><name>{escape(Path(m["audio"]).name)}</name><duration>{adur}</duration>{rate}'
             f'<start>0</start><end>{total}</end><in>0</in><out>{total}</out>'
             f'<file id="file-music"><name>{escape(Path(m["audio"]).name)}</name>'
             f'<pathurl>file://localhost{quote(m["audio"])}</pathurl>{rate}<duration>{adur}</duration>'
             f'<media><audio><channelcount>2</channelcount></audio></media></file>'
             f'<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack></clipitem>')
    marks = "".join(f'<marker><name>{escape(k["name"])}</name><comment>{escape(k["kind"] + "：" + k["note"])}</comment>'
                    f'<in>{k["frame"]}</in><out>-1</out></marker>' for k in mk)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="5"><sequence id="seq-1">'
            f'<name>小耳音乐卡点MV · {escape(Path(m["audio"]).stem)}</name><duration>{total}</duration>{rate}'
            f'<timecode>{rate}<string>00:00:00:00</string><frame>0</frame><displayformat>NDF</displayformat></timecode>'
            f'<media><video><format><samplecharacteristics>{rate}<width>{W}</width><height>{H}</height>'
            f'<pixelaspectratio>square</pixelaspectratio></samplecharacteristics></format>'
            f'<track>{"".join(items)}</track></video><audio><track>{audio}</track></audio></media>{marks}</sequence></xmeml>\n')


def to_fcpxml(plan, cues, mk, with_speed):
    m = plan["meta"]
    fps = int(round(m["fps"]))
    W, H = m["size"].split("x")
    total = plan["shots"][-1]["frame_out"]

    def T(fr):
        return f"{int(fr)}/{fps}s" if fr else "0s"
    assets, ids = [], {}
    adur = int(round(cues["meta"]["duration"] * fps))
    assets.append(f'<asset id="a0" name="{escape(Path(m["audio"]).name)}" start="0s" duration="{T(adur)}" hasAudio="1" '
                  f'audioSources="1" audioChannels="2"><media-rep kind="original-media" src="file://{quote(m["audio"])}"/></asset>')
    for s in plan["shots"]:
        p = s["clip"]
        if p not in ids:
            ids[p] = f"a{len(ids) + 1}"
            d = dur_frames(p, fps)
            assets.append(f'<asset id="{ids[p]}" name="{escape(Path(p).name)}" start="0s" duration="{T(d)}" hasVideo="1" '
                          f'format="r1"><media-rep kind="original-media" src="file://{quote(str(Path(p).resolve()))}"/></asset>')
    clips = []
    for s in plan["shots"]:
        a, b = src_range(s, fps, with_speed)
        speed = ""
        if with_speed and abs(s["speed"] - 1) > 1e-3:
            speed = (f'<timeMap><timept time="0s" value="{T(a)}" interp="linear"/>'
                     f'<timept time="{T(s["frames"])}" value="{T(b)}" interp="linear"/></timeMap>')
        clips.append(f'<asset-clip ref="{ids[s["clip"]]}" lane="1" offset="{T(s["frame_in"])}" name="{s["i"]:03d} {escape(Path(s["clip"]).stem)}" '
                     f'start="{T(a)}" duration="{T(s["frames"])}">{speed}<note>{escape(shot_note(s))}</note></asset-clip>')
    marks = "".join(f'<marker start="{T(k["frame"])}" duration="1/{fps}s" value="{escape(k["name"])}" note="{escape(k["note"])}"/>' for k in mk)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n<fcpxml version="1.10"><resources>'
            f'<format id="r1" name="FFVideoFormat{H}p{fps}" frameDuration="1/{fps}s" width="{W}" height="{H}"/>{"".join(assets)}</resources>'
            f'<library><event name="小耳音乐卡点MV"><project name="小耳音乐卡点MV · {escape(Path(m["audio"]).stem)}">'
            f'<sequence format="r1" duration="{T(total)}" tcStart="0s" tcFormat="NDF"><spine>'
            f'<asset-clip ref="a0" offset="0s" name="{escape(Path(m["audio"]).name)}" start="0s" duration="{T(total)}">'
            f'{marks}{"".join(clips)}</asset-clip></spine></sequence></project></event></library></fcpxml>\n')


def to_csv(plan, mk, path):
    fps = plan["meta"]["fps"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["时间码", "秒", "帧", "类型", "名称", "说明"])
        for k in mk:
            w.writerow([tc(k["frame"], fps), f"{k['frame'] / fps:.3f}", k["frame"], k["kind"], k["name"], k["note"]])
        for s in plan["shots"]:
            w.writerow([tc(s["frame_in"], fps), f"{s['frame_in'] / fps:.3f}", s["frame_in"], "切点",
                        f"镜头 {s['i']} · {Path(s['clip']).name}", shot_note(s)])


def main(argv=None):
    ap = argparse.ArgumentParser(description="小耳音乐卡点MV · 导出到剪辑软件")
    ap.add_argument("plan")
    ap.add_argument("--format", choices=["xml", "fcpxml", "csv", "all"], default="all")
    ap.add_argument("-o", "--out", help="输出文件前缀（默认 plan 同目录的 timeline）")
    ap.add_argument("--with-speed", action="store_true", help="把方案里的变速写进工程")
    ap.add_argument("--markers", choices=["structure", "bars", "beats"], default="bars")
    a = ap.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text())
    if abs(plan["meta"]["fps"] - round(plan["meta"]["fps"])) > 1e-6:
        raise SystemExit("✗ 目前只支持整数帧率（24/25/30/50/60）")
    cues = json.loads(Path(plan["meta"]["cues"]).read_text())
    mk = markers(plan, cues, a.markers)
    pre = Path(a.out or Path(a.plan).with_name("timeline"))
    done = []
    if a.format in ("xml", "all"):
        p = pre.parent / (pre.name + ".xml"); p.write_text(to_xmeml(plan, cues, mk, a.with_speed)); done.append(f"{p}（Premiere / Resolve）")
    if a.format in ("fcpxml", "all"):
        p = pre.parent / (pre.name + ".fcpxml"); p.write_text(to_fcpxml(plan, cues, mk, a.with_speed)); done.append(f"{p}（Final Cut / Resolve）")
    if a.format in ("csv", "all"):
        p = pre.with_name(pre.name + "-markers.csv"); to_csv(plan, mk, p); done.append(f"{p}（标记清单）")
    print(f"✓ {len(plan['shots'])} 个镜头 · {len(mk)} 个标记")
    for d in done:
        print("  " + d)


if __name__ == "__main__":
    main()
