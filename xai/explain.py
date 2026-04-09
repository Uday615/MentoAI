"""
XAI Explanation Module - AI Mentor Decision Support System
Uses SHAP to generate feature attribution explanations.
Provides domain-level and question-level decomposition.
"""
import numpy as np
import shap
import pickle
from typing import Dict, List, Tuple, Optional


# ────────────────────────────────────────────────────────────────
# Domain Definitions (mirrors ml/train.py)
# ────────────────────────────────────────────────────────────────
DOMAIN_MAP = {
    "Academic Stress": [0, 1, 2, 3, 4],
    "Sleep Disruption": [5, 6, 7],
    "Emotional Exhaustion": [8, 9, 10],
    "Motivation Decline": [11, 12, 13],
    "Cognitive Overload": [14, 15, 16],
    "Behavioral Withdrawal": [17, 18, 19, 20],
    "Support Availability": [21, 22, 23, 24],
}

DOMAIN_LABELS = {
    "Academic Stress": "📚 Academic Stress",
    "Sleep Disruption": "😴 Sleep Disruption",
    "Emotional Exhaustion": "🔥 Emotional Exhaustion",
    "Motivation Decline": "📉 Motivation Decline",
    "Cognitive Overload": "🧠 Cognitive Overload",
    "Behavioral Withdrawal": "🚪 Behavioral Withdrawal",
    "Support Availability": "🤝 Support Availability",
}


class MentorExplainer:
    """
    SHAP-based explainer for mentor decision support.
    Works with Random Forest and Gradient Boosting (TreeExplainer).
    Falls back to KernelExplainer for Logistic Regression.
    """

    def __init__(self, model, feature_names: List[str], background_data=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

        # Determine explainer type
        model_type = type(model).__name__
        inner_model = model

        # If Pipeline, extract the classifier
        if hasattr(model, 'named_steps'):
            inner_model = list(model.named_steps.values())[-1]
            model_type = type(inner_model).__name__

        try:
            if model_type in ("RandomForestClassifier", "GradientBoostingClassifier",
                              "DecisionTreeClassifier", "ExtraTreesClassifier"):
                self.explainer = shap.TreeExplainer(inner_model)
                self._explainer_type = "tree"
            else:
                # Logistic Regression — use LinearExplainer
                self.explainer = shap.LinearExplainer(inner_model, background_data)
                self._explainer_type = "linear"
        except Exception as e:
            print(f"[XAI] Explainer init warning: {e}")
            self.explainer = None
            self._explainer_type = None

    def explain(self, X_input: np.ndarray, predicted_class: int) -> Dict:
        """
        Generate SHAP explanation for a single prediction.
        Returns structured dict with feature attributions and domain scores.
        """
        if self.explainer is None:
            return self._fallback_explanation(X_input, predicted_class)

        try:
            shap_values = self.explainer.shap_values(X_input)

            # Handle both list (multiclass) and array outputs
            if isinstance(shap_values, list):
                # List of arrays, one per class
                class_shap = shap_values[predicted_class][0]
            elif shap_values.ndim == 3:
                # Shape: (samples, features, classes)
                class_shap = shap_values[0, :, predicted_class]
            else:
                class_shap = shap_values[0]

            return self._build_explanation(X_input[0], class_shap, predicted_class)

        except Exception as e:
            print(f"[XAI] SHAP computation error: {e}")
            return self._fallback_explanation(X_input, predicted_class)

    def _build_explanation(self, input_values: np.ndarray,
                           shap_vals: np.ndarray,
                           predicted_class: int) -> Dict:
        """Build comprehensive explanation dictionary."""

        n = len(self.feature_names)

        # ── Per-feature attributions ──────────────────────────────
        feature_attributions = []
        for i in range(n):
            feature_attributions.append({
                "feature": self.feature_names[i],
                "input_value": int(input_values[i]),
                "shap_value": round(float(shap_vals[i]), 4),
                "direction": "positive" if shap_vals[i] > 0 else "negative",
            })

        # Sort by absolute SHAP value
        feature_attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # ── Top-N key indicators (positive SHAP driving assessed class) ──
        positive_contributors = [
            f for f in feature_attributions if f["shap_value"] > 0
        ]
        top_indicators = [f["feature"] for f in positive_contributors[:5]]
        if not top_indicators:
            top_indicators = [feature_attributions[0]["feature"]] if feature_attributions else ["General indicators"]

        # ── Domain-level scores ───────────────────────────────────
        domain_scores = {}
        for domain, indices in DOMAIN_MAP.items():
            valid_indices = [i for i in indices if i < n]
            if not valid_indices:
                continue
            domain_shap = np.sum([shap_vals[i] for i in valid_indices])
            domain_avg_input = np.mean([input_values[i] for i in valid_indices])
            domain_scores[domain] = {
                "label": DOMAIN_LABELS.get(domain, domain),
                "shap_sum": round(float(domain_shap), 4),
                "avg_input": round(float(domain_avg_input), 2),
                "risk_contribution": round(float(domain_shap), 4),
            }

        # Sort domains by contribution
        ranked_domains = sorted(
            domain_scores.items(),
            key=lambda x: x[1]["shap_sum"],
            reverse=True
        )

        # ── Validation flag ───────────────────────────────────────
        consistency_check = self._validate_explanation(
            input_values, shap_vals, predicted_class, domain_scores
        )

        return {
            "top_indicators": top_indicators[:3],
            "feature_attributions": feature_attributions[:10],  # top 10
            "domain_scores": dict(ranked_domains),
            "all_shap_values": [round(float(v), 4) for v in shap_vals],
            "consistency_check": consistency_check,
        }

    def _validate_explanation(self, input_values, shap_vals, predicted_class, domain_scores) -> Dict:
        """
        Validate that XAI output is consistent with rule-based expectations.
        Detects anomalies between SHAP explanation and intuitive scoring.
        """
        warnings = []
        is_consistent = True

        # Check: if high risk predicted, at least one high domain should be active
        if predicted_class == 2:  # High Risk
            high_scoring_features = [
                input_values[i] for i in range(len(input_values)) if input_values[i] >= 4
            ]
            if len(high_scoring_features) < 3:
                warnings.append("High risk predicted but few high-scoring features — review carefully.")
                is_consistent = False

        # Check: if low risk predicted, no extreme scores should exist
        if predicted_class == 0:  # Low Risk
            extreme_features = [i for i in range(len(input_values)) if input_values[i] == 5]
            if len(extreme_features) >= 5:
                warnings.append("Low risk predicted but several extreme (5) scores found — verify.")
                is_consistent = False

        return {
            "is_consistent": is_consistent,
            "warnings": warnings,
        }

    def _fallback_explanation(self, X_input: np.ndarray, predicted_class: int) -> Dict:
        """Fallback when SHAP fails — use simple feature value ranking."""
        input_vals = X_input[0]
        indices = np.argsort(input_vals)[::-1]
        top_indicators = [self.feature_names[i] for i in indices[:3]]

        return {
            "top_indicators": top_indicators,
            "feature_attributions": [],
            "domain_scores": {},
            "all_shap_values": [],
            "consistency_check": {"is_consistent": True, "warnings": ["SHAP unavailable; using raw scores."]},
        }


def build_explainer_from_pkl(model_pkl_path: str) -> Optional[MentorExplainer]:
    """Load model from pickle and construct MentorExplainer."""
    try:
        with open(model_pkl_path, "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        features = data["features"]

        # Generate dummy background data for LinearExplainer
        background = np.full((1, len(features)), 3.0)  # neutral baseline

        return MentorExplainer(model, features, background_data=background)
    except Exception as e:
        print(f"[XAI] Failed to build explainer: {e}")
        return None
