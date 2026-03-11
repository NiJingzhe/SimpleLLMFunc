#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

poetry run python scripts/translate_docs_locale.py \
  --target-lang en \
  --source-lang zh_CN \
  --translate \
  --include-fuzzy \
  --normalize-metadata \
  "$@"
