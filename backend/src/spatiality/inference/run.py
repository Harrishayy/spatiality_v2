"""Inference entrypoint — stub.

Called from ``modal_inference.py::run_inference_one``. Replace the body with
the actual inference logic once the model is chosen.
"""

from __future__ import annotations


def run(input_id: str, **kwargs) -> dict:
    return {"input_id": input_id, "status": "stub", "kwargs": kwargs}
