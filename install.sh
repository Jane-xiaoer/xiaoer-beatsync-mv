#!/usr/bin/env bash
# 小耳音乐卡点MV · 一键安装：建 Python 环境、装依赖（Demucs / librosa / OpenCV），检查 ffmpeg
set -e
D="$(cd "$(dirname "$0")" && pwd)"
command -v ffmpeg >/dev/null || { echo "✗ 需要 ffmpeg：macOS 上 brew install ffmpeg"; exit 1; }
if command -v uv >/dev/null; then
  uv venv -q "$D/.venv" --python 3.11
  uv pip install -q --python "$D/.venv/bin/python" -r "$D/requirements.txt"
else
  python3 -m venv "$D/.venv"
  "$D/.venv/bin/pip" install -q -r "$D/requirements.txt"
fi
"$D/.venv/bin/python" -c "import librosa, demucs, cv2, torch; print('✓ 环境就绪 · GPU(MPS):', torch.backends.mps.is_available())"
echo "试一下：$D/beatsync.sh --help"
