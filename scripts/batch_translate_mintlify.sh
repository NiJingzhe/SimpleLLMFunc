#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

poetry run python scripts/translate_mintlify_docs.py \
  --target-lang en \
  --source-lang zh_CN \
  --sync-pages \
  --translate \
  "$@"
