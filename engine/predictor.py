from pathlib import Path
import joblib

class Predictor:

    def __init__(self):

        model_path = (
            Path(__file__).resolve().parent.parent
            / "Backtest"
            / "xgboost_vix_best_model.pkl"
        )

        print("Loading model:", model_path)

        self.model = joblib.load(model_path)

        try:
            self.expected_features = list(
                self.model.feature_names_in_
            )
        except Exception:
            self.expected_features = None

    def predict(self, feature_df):

        latest = feature_df.tail(1)

        if self.expected_features:
            latest = latest[self.expected_features]

        prob = self.model.predict_proba(latest)[:, 1]

        return float(prob[0])
