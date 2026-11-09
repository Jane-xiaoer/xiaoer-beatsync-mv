#!/usr/bin/env bash
# 小耳音乐卡点MV命令入口：自动用项目自带的 Python 环境
D="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec "$D/.venv/bin/python" "$D/beatsync/__main__.py" "$@"
