import json
import os
from typing import Dict, List


def load_calibration_jsonl(path: str, text_key: str = "text") -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise ValueError(f"Calibration generation: jsonl not found: {path}")

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Calibration generation: invalid jsonl at line {line_no}") from exc

            text = obj.get(text_key) if isinstance(obj, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue

            rows.append({"text": text})

    if not rows:
        raise ValueError("Calibration generation: jsonl is empty or has no valid text rows.")

    return rows