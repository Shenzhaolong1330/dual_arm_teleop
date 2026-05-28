#!/usr/bin/env bash
set -euo pipefail

# 清理 Python/构建产生的临时文件（不会删除源码）
# 用法：在项目任意目录执行 `./rm_tmp.sh`

# 定位到脚本所在目录（通常是项目根目录）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[clean] root: $ROOT_DIR"

# 常见构建产物/缓存目录（递归清理当前目录及所有子目录）
build_patterns=(
  build
  "*.dist-info"
  dist
  "*.egg-info"
  .eggs
  .pytest_cache
  .mypy_cache
  .ruff_cache
  .tox
  htmlcov
  pip-wheel-metadata
  __pypackages__
)

for pattern in "${build_patterns[@]}"; do
  find "$ROOT_DIR" -type d -name "$pattern" -prune -exec rm -rf -- {} + 2>/dev/null || true
 done

find "$ROOT_DIR" -type f \( -name ".coverage" -o -name ".coverage.*" \) -delete 2>/dev/null || true

# Python bytecode / cache
find "$ROOT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true

echo "[clean] done"
