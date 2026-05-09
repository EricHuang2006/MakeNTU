from config import HAND_SIGN_MODEL_PATH
from event_logger import log_event


class HandSignClassifier:
    def __init__(self, model_path=HAND_SIGN_MODEL_PATH):
        self.model_path = model_path
        self.available = False
        log_event(
            "system",
            (
                "Hand sign classifier is using abstract placeholder API; "
                f"model_path={self.model_path}"
            ),
            throttle_seconds=0.0,
        )

    def classify(self, _frame):
        return None

    def close(self):
        pass
