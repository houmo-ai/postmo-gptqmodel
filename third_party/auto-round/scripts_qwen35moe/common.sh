#!/bin/bash

_python_cuda_status() {
    local python_bin="$1"
    "${python_bin}" - <<'PY'
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - shell helper
    print(f"import_failed:{exc}")
    sys.exit(1)

print(f"torch={torch.__version__},cuda_available={torch.cuda.is_available()}")
PY
}


_python_has_cuda() {
    local python_bin="$1"
    "${python_bin}" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}


resolve_qwen35moe_python() {
    local preferred="${AR_PYTHON:-python}"
    if _python_has_cuda "${preferred}"; then
        printf '%s\n' "${preferred}"
        return 0
    fi

    if [ -n "${AR_PYTHON:-}" ]; then
        echo "ERROR: AR_PYTHON=${preferred} does not have a usable CUDA runtime." >&2
        echo "       Status: $(_python_cuda_status "${preferred}" 2>/dev/null || echo unavailable)" >&2
        return 1
    fi

    local -a fallbacks=()
    if [ -n "${AR_FALLBACK_PYTHON:-}" ]; then
        fallbacks+=("${AR_FALLBACK_PYTHON}")
    fi

    if [ -n "${CONDA_EXE:-}" ]; then
        local conda_root
        conda_root="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
        if [ -x "${conda_root}/envs/xhquant/bin/python" ]; then
            fallbacks+=("${conda_root}/envs/xhquant/bin/python")
        fi
    fi

    local candidate
    for candidate in "${fallbacks[@]}"; do
        if [ "${candidate}" = "${preferred}" ]; then
            continue
        fi
        if _python_has_cuda "${candidate}"; then
            echo "[info] Current python lacks usable CUDA ($(_python_cuda_status "${preferred}" 2>/dev/null || echo unavailable));" >&2
            echo "[info] Falling back to ${candidate}" >&2
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    echo "ERROR: Current python does not have a usable CUDA runtime." >&2
    echo "       Status: $(_python_cuda_status "${preferred}" 2>/dev/null || echo unavailable)" >&2
    echo "       Set AR_PYTHON or AR_FALLBACK_PYTHON to a CUDA-capable interpreter." >&2
    return 1
}
