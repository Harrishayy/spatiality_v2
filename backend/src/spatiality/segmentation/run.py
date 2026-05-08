"""Segmentation entrypoint — stub.

Called from ``modal_segmentation.py::run_segmentation_one``. Replace the body
with the actual segmentation logic once the model is chosen.
"""

from __future__ import annotations


def run(input_id: str, **kwargs) -> dict:
    return {"input_id": input_id, "status": "stub", "kwargs": kwargs}
