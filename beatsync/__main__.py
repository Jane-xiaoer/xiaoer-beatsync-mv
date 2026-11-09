"""小耳音乐卡点MV beatsync —— 给一段音乐，做一支卡点精确到帧的 MV。

  beatsync stems song.mp3                分轨：Demucs 4 轨（鼓/贝斯/人声/其他），--six 拆 6 轨
  beatsync analyze song.mp3              听歌：先分轨，再找拍 / 小节 / 乐句 / 段落 / 各轨起音（带置信度）
  beatsync kells song.cues.json 片段…    ★ 凯尔经模式：一组生长/变形片段 + 歌 → 音乐推着画面长的 MV（自动编排 + 推播放头）
  beatsync plan song.cues.json clips/    剪辑模式：普通素材按三层规则排出剪辑方案（每刀写明为什么）
  beatsync render plan.json              出片：直接渲染 MV
  beatsync export plan.json              导出：Premiere / Resolve（XML）、Final Cut（FCPXML）、标记清单（CSV）
  beatsync check mv.mp4 --plan plan.json 体检：从成片逐帧量每一刀差几帧
  beatsync make song.mp3 clips/          一条龙：analyze → plan → render → export → check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def make(argv):
    import argparse
    import analyze, plan, render, export, check
    ap = argparse.ArgumentParser(prog="beatsync make", description="一条龙出片")
    ap.add_argument("audio")
    ap.add_argument("clips", nargs="+")
    ap.add_argument("-o", "--outdir")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--density", default="standard", choices=["calm", "standard", "energetic"])
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--six", action="store_true", help="分 6 轨")
    ap.add_argument("--no-stems", action="store_true", help="不分轨（快，但卡点单薄）")
    a = ap.parse_args(argv)
    out = Path(a.outdir or Path(a.audio).with_suffix("").name + "-卡点")
    out.mkdir(parents=True, exist_ok=True)
    cues = out / "cues.json"
    print("① 分轨 + 听歌"); analyze.main([a.audio, "-o", str(cues), "--fps", str(a.fps)]
                                        + (["--six"] if a.six else []) + (["--no-stems"] if a.no_stems else []))
    print("② 编排"); plan.main([str(cues), *a.clips, "--audio", a.audio, "-o", str(out / "plan.json"),
                               "--density", a.density, "--size", a.size])
    print("③ 出片"); render.main([str(out / "plan.json"), "-o", str(out / "mv.mp4")] + (["--preview"] if a.preview else []))
    print("④ 导出"); export.main([str(out / "plan.json"), "-o", str(out / "timeline")])
    print("⑤ 体检"); check.main([str(out / "mv.mp4"), "--plan", str(out / "plan.json"), "-o", str(out / "卡点体检.html")])


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "make":
        return make(rest)
    mods = {"kells": "kells", "grow": "grow", "stems": "stems", "analyze": "analyze", "plan": "plan", "render": "render", "export": "export", "check": "check"}
    if cmd not in mods:
        print(f"未知命令 {cmd}\n{__doc__}"); sys.exit(2)
    __import__(mods[cmd]).main(rest)


if __name__ == "__main__":
    main()
