from typing import Any, Dict


class AnalysisService:

    def __init__(self) -> None:
        self._ready = False  # поки без ML

    def is_ready(self) -> bool:
        return self._ready

    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        return {
            "error": "Analysis service is not connected yet",
            "message": "ML engine will be added in the next step"
        }