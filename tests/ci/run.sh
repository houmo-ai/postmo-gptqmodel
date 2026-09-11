#!/usr/bin/env bash
set -euo pipefail
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "${ci_dir}/run.py" "$@"
