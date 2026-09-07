"""Run the maintained Qwen3-Next AutoRound backend with vendored AutoRound."""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    if "--deterministic" in sys.argv:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "third_party/auto-round"))
    from gptqmodel.recipes.qwen3_next_autoround import main

    main()
