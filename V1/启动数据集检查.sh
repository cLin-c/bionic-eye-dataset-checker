#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$APP_DIR"
PYTHON_BIN="${DATASET_EDITOR_PYTHON:-python3}"
if ! "$PYTHON_BIN" -c 'import PyQt5.QtWidgets, yaml' >/dev/null 2>&1; then
    echo '当前 Python 缺少运行依赖，请执行：'
    echo "$PYTHON_BIN -m pip install -r \"$APP_DIR/requirements.txt\""
    echo '也可设置 DATASET_EDITOR_PYTHON 为已安装依赖的 Python 路径。'
    exit 1
fi
exec "$PYTHON_BIN" "$APP_DIR/app.py" "$@"
