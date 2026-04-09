"""
FastAPI Backend - AI Mentor Decision Support System

Endpoints:
  GET  /                          - Health check
  POST /api/screen                - Stage 1: 10-question weekly screening
  POST /api/predict               - Stage 2: Full 25-question assessment + ML + XAI + RAG
  GET  /api/models/metrics        - ML model performance metrics
  GET  /api/config/questions      - All question definitions
"""

import json
import os
import sys
import pickle
import re
import smtplib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from email.mime.text import MIMEText

# â”€â”€ Self-Healing Path Injection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If running with system Python, try to include the project's venv
try:
    import passlib
except ImportError:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(backend_dir)
    venv_site_packages = os.path.join(root_dir, "venv", "Lib", "site-packages")
    if os.path.exists(venv_site_packages):
        sys.path.insert(0, venv_site_packages)
        # Note: We need to try imports again or handle them lazily
    else:
        print(f"[Backend] WARNING: passlib not found and venv not at {venv_site_packages}")
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import bcrypt

# MongoDB / In-Memory Storage
try:
    from database import (
        save_screening_session,
        save_assessment_session,
        get_recent_assessments,
        get_student_history,
        get_feedback_history,
        create_feedback_request,
        get_feedback_request_by_token,
        get_db_stats,
        is_connected,
        using_memory,
        create_user,
        get_user_by_email,
        complete_feedback_request,
        save_feedback_session,
    )
    _MONGO_AVAILABLE = True
except ImportError as e:
    try:
        from backend.database import (
            save_screening_session,
            save_assessment_session,
            get_recent_assessments,
            get_student_history,
            get_feedback_history,
            create_feedback_request,
            get_feedback_request_by_token,
            get_db_stats,
            is_connected,
            using_memory,
            create_user,
            get_user_by_email,
            complete_feedback_request,
            save_feedback_session,
        )
        _MONGO_AVAILABLE = True
    except ImportError:
        print(f"[Backend] Database module not found: {e}")
        _MONGO_AVAILABLE = False

# â”€â”€ Path setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BACKEND_DIR)
ML_DIR      = os.path.join(ROOT_DIR, "ml")
RAG_DIR     = os.path.join(ROOT_DIR, "rag")
XAI_DIR     = os.path.join(ROOT_DIR, "xai")

for d in [ROOT_DIR, ML_DIR, XAI_DIR, RAG_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = FastAPI(
    title="AI Mentor Decision Support System",
    description="Hybrid ML + XAI + RAG backend for early student distress identification.",
    version="2.1.0",
)

FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "http://localhost:5173")
GMAIL_SENDER = os.environ.get("GMAIL_SENDER_EMAIL", "").strip()
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
GOOGLE_FEEDBACK_FORM_URL = os.environ.get(
    "GOOGLE_FEEDBACK_FORM_URL",
    "https://forms.gle/tEtMcqU32yDQL35s5",
).strip()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Question Definitions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Stage 1: 10-question Weekly Screening (subset of 25)
SCREENING_QUESTIONS = [
    {"id": "sq1",  "text": "The student appears stressed or emotionally irritated.",          "domain": "Emotional Exhaustion"},
    {"id": "sq2",  "text": "The student shows signs of low motivation or disengagement.",     "domain": "Motivation Decline"},
    {"id": "sq3",  "text": "The student seems withdrawn or avoids peer interaction.",         "domain": "Behavioral Withdrawal"},
    {"id": "sq4",  "text": "The student has missed recent classes or scheduled sessions.",    "domain": "Behavioral Withdrawal"},
    {"id": "sq5",  "text": "The student appears fatigued or has low energy consistently.",    "domain": "Sleep Disruption"},
    {"id": "sq6",  "text": "The student has expressed worry, anxiety, or hopelessness.",      "domain": "Emotional Exhaustion"},
    {"id": "sq7",  "text": "The student is struggling with academic coursework or deadlines.","domain": "Academic Stress"},
    {"id": "sq8",  "text": "The student has reduced communication with me as their mentor.",  "domain": "Behavioral Withdrawal"},
    {"id": "sq9",  "text": "The student seems to lack confidence in their academic ability.", "domain": "Motivation Decline"},
    {"id": "sq10", "text": "The student mentions lacking support from family or friends.",    "domain": "Support Availability"},
]

# Stage 2: Full 25-question Detailed Assessment
DETAILED_QUESTIONS = [
    # Academic Stress (Q1-Q5)
    {"id": "q1",  "text": "The student struggles significantly with academic coursework.",       "domain": "Academic Stress"},
    {"id": "q2",  "text": "The student appears overwhelmed by responsibilities.",                "domain": "Academic Stress"},
    {"id": "q3",  "text": "The student frequently requests extensions for deadlines.",           "domain": "Academic Stress"},
    {"id": "q4",  "text": "The student shows inconsistency in assignment submission.",           "domain": "Academic Stress"},
    {"id": "q5",  "text": "There has been a noticeable drop in the student's grades.",          "domain": "Academic Stress"},
    # Sleep Disruption (Q6-Q8)
    {"id": "q6",  "text": "The student consistently shows fatigue or low energy in sessions.",  "domain": "Sleep Disruption"},
    {"id": "q7",  "text": "The student shows a visible decline in physical self-presentation.", "domain": "Sleep Disruption"},
    {"id": "q8",  "text": "The student reports or shows poor balance across life activities.",  "domain": "Sleep Disruption"},
    # Emotional Exhaustion (Q9-Q11)
    {"id": "q9",  "text": "The student frequently appears stressed, tense, or irritated.",      "domain": "Emotional Exhaustion"},
    {"id": "q10", "text": "The student expresses anxiety or worry on a daily basis.",           "domain": "Emotional Exhaustion"},
    {"id": "q11", "text": "The student has expressed feelings of hopelessness or despair.",     "domain": "Emotional Exhaustion"},
    # Motivation Decline (Q12-Q14)
    {"id": "q12", "text": "The student shows low enthusiasm for learning activities.",          "domain": "Motivation Decline"},
    {"id": "q13", "text": "The student demonstrates low motivation to achieve their goals.",    "domain": "Motivation Decline"},
    {"id": "q14", "text": "The student lacks confidence in their academic subject areas.",      "domain": "Motivation Decline"},
    # Cognitive Overload (Q15-Q17)
    {"id": "q15", "text": "The student shows signs of overthinking or cognitive rumination.",   "domain": "Cognitive Overload"},
    {"id": "q16", "text": "The student procrastinates on important academic tasks.",            "domain": "Cognitive Overload"},
    {"id": "q17", "text": "The student is nervous or anxious when speaking publicly.",         "domain": "Cognitive Overload"},
    # Behavioral Withdrawal (Q18-Q21)
    {"id": "q18", "text": "The student is quiet or withdrawn in peer settings.",               "domain": "Behavioral Withdrawal"},
    {"id": "q19", "text": "The student has reduced communication with their mentor.",          "domain": "Behavioral Withdrawal"},
    {"id": "q20", "text": "The student rarely initiates conversations or interactions.",       "domain": "Behavioral Withdrawal"},
    {"id": "q21", "text": "The student misses classes or scheduled sessions frequently.",      "domain": "Behavioral Withdrawal"},
    # Support Availability (Q22-Q25)
    {"id": "q22", "text": "The student lacks visible support from family or close friends.",   "domain": "Support Availability"},
    {"id": "q23", "text": "The student spends excessive time on social media/digital devices.","domain": "Support Availability"},
    {"id": "q24", "text": "The student displays poor behaviour in group or social settings.",  "domain": "Support Availability"},
    {"id": "q25", "text": "The student shows low confidence in overcoming personal challenges.","domain": "Support Availability"},
]

DOMAIN_MAP = {
    "Academic Stress":       [0, 1, 2, 3, 4],
    "Sleep Disruption":      [5, 6, 7],
    "Emotional Exhaustion":  [8, 9, 10],
    "Motivation Decline":    [11, 12, 13],
    "Cognitive Overload":    [14, 15, 16],
    "Behavioral Withdrawal": [17, 18, 19, 20],
    "Support Availability":  [21, 22, 23, 24],
}

FEATURE_NAMES = [q["text"] for q in DETAILED_QUESTIONS]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Model & Explainer Loading
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model = None
explainer = None
rag_engine = None
model_metadata = {}

def load_model():
    global model, explainer, rag_engine, model_metadata
    model_path = os.path.join(ML_DIR, "model.pkl")
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        model_metadata = {
            "model_name": data.get("model_name", "Unknown"),
            "features": data.get("features", FEATURE_NAMES),
        }
        print(f"[Backend] ML Model loaded: {model_metadata['model_name']}")
    except Exception as e:
        print(f"[Backend] WARNING: ML Model not loaded â€” {e}")
        model = None

    # Load XAI explainer
    try:
        from xai.explain import MentorExplainer
        if model is not None:
            background = np.full((1, 25), 3.0)
            explainer = MentorExplainer(model, FEATURE_NAMES, background_data=background)
            print("[Backend] XAI Explainer ready.")
    except Exception as e:
        print(f"[Backend] WARNING: XAI explainer not loaded â€” {e}")
        explainer = None

    # Load RAG engine
    try:
        from rag.generator import MentorRAG
        vector_db_path = os.path.join(RAG_DIR, "vector_db")
        rag_engine = MentorRAG(vector_db_path=vector_db_path)
        print("[Backend] RAG Engine ready.")
    except Exception as e:
        print(f"[Backend] WARNING: RAG engine not loaded â€” {e}")
        rag_engine = None


load_model()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Rule-Based Hybrid Layer
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DOMAIN_WEIGHTS = {
    "Academic Stress":       1.2,
    "Sleep Disruption":      1.0,
    "Emotional Exhaustion":  1.8,
    "Motivation Decline":    1.3,
    "Cognitive Overload":    1.0,
    "Behavioral Withdrawal": 1.5,
    "Support Availability":  1.1,
}

def rule_based_summary(answers: List[int]) -> Dict:
    """Return weighted score details used by the rule-based layer."""
    weighted_total = 0.0
    max_possible = 0.0

    for domain, indices in DOMAIN_MAP.items():
        valid = [answers[i] for i in indices if i < len(answers)]
        if not valid:
            continue
        avg = sum(valid) / len(valid)
        weight = DOMAIN_WEIGHTS[domain]
        weighted_total += avg * weight
        max_possible += 5 * weight

    ratio = weighted_total / max_possible if max_possible > 0 else 0.0
    return {
        "weighted_score": round(weighted_total, 4),
        "max_weighted_score": round(max_possible, 4),
        "score_ratio": round(ratio, 4),
    }

def rule_based_classify(answers: List[int]) -> int:
    """Weighted domain rule-based classifier. Returns 0=Low, 1=Moderate, 2=High."""
    weighted_total = 0.0
    max_possible   = 0.0

    for domain, indices in DOMAIN_MAP.items():
        valid = [answers[i] for i in indices if i < len(answers)]
        if not valid:
            continue
        avg = sum(valid) / len(valid)
        w = DOMAIN_WEIGHTS[domain]
        weighted_total += avg * w
        max_possible   += 5 * w

    # Critical override: hopelessness (index 10) at max â†’ always High
    if len(answers) > 10 and answers[10] == 5:
        return 2

    ratio = weighted_total / max_possible if max_possible > 0 else 0
    if ratio <= 0.47:
        return 0
    elif ratio <= 0.68:
        return 1
    else:
        return 2

def hybrid_classify(answers: List[int], ml_pred: int) -> Dict:
    """
    Combine rule-based and ML predictions.
    Returns final decision with conflict resolution metadata.
    """
    rule_pred = rule_based_classify(answers)
    label_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
    conflict = rule_pred != ml_pred
    score_summary = rule_based_summary(answers)
    score_ratio = score_summary["score_ratio"]

    if not conflict:
        final = ml_pred
        resolution = "Both rule-based and ML models agree."
    else:
        if rule_pred == 0 and ml_pred == 1 and score_ratio <= 0.42:
            final = rule_pred
            resolution = (
                f"Rule-based predicted {label_map[rule_pred]}, ML predicted {label_map[ml_pred]}. "
                "Weighted score is clearly within the low-risk band, so the final class remains Low Risk."
            )
        elif rule_pred == 2 and ml_pred == 1 and score_ratio >= 0.72:
            final = rule_pred
            resolution = (
                f"Rule-based predicted {label_map[rule_pred]}, ML predicted {label_map[ml_pred]}. "
                "Weighted score is clearly within the high-risk band, so the final class remains High Risk."
            )
        else:
            final = max(rule_pred, ml_pred)
            resolution = (
                f"Rule-based predicted {label_map[rule_pred]}, ML predicted {label_map[ml_pred]}. "
                f"Conservative conflict resolution: using higher risk ({label_map[final]})."
            )

    return {
        "final_class":    final,
        "rule_class":     rule_pred,
        "ml_class":       ml_pred,
        "conflict":       conflict,
        "resolution":     resolution,
        "rule_label":     label_map[rule_pred],
        "ml_label":       label_map[ml_pred],
        "final_label":    label_map[final],
        **score_summary,
    }

def screen_classify(answers: List[int]) -> Dict:
    """Stage 1 screening classifier based on 10 screening questions."""
    total = sum(answers)
    # Max = 50, mapping: â‰¤22 Low, â‰¤35 Moderate, >35 High
    if total <= 22:
        risk_class = 0
    elif total <= 35:
        risk_class = 1
    else:
        risk_class = 2

    label_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
    proceed = risk_class >= 1  # Recommend detailed assessment if not Low

    return {
        "risk_class": risk_class,
        "risk_label": label_map[risk_class],
        "total_score": total,
        "max_score":   50,
        "proceed_to_stage2": proceed,
        "triage_message": (
            "Proceed to detailed 25-item assessment for a full ML-based analysis."
            if proceed else
            "No immediate intervention required. Repeat screening next week."
        ),
    }

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pydantic Schemas
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class ScreeningInput(BaseModel):
    student_name: Optional[str] = "Anonymous"
    student_id:   Optional[str] = ""
    answers:      List[int]

    @field_validator("answers")
    @classmethod
    def validate_answers(cls, v):
        if len(v) != 10:
            raise ValueError(f"Stage 1 expects 10 answers, got {len(v)}")
        if any(a < 1 or a > 5 for a in v):
            raise ValueError("All answers must be between 1 and 5")
        return v

class AssessmentInput(BaseModel):
    student_name: Optional[str] = "Anonymous"
    student_id:   Optional[str] = ""
    programme:    Optional[str] = ""
    answers:      List[int]

    @field_validator("answers")
    @classmethod
    def validate_answers(cls, v):
        if len(v) != 25:
            raise ValueError(f"Stage 2 expects 25 answers, got {len(v)}")
        if any(a < 1 or a > 5 for a in v):
            raise ValueError("All answers must be between 1 and 5")
        return v

class UserSignup(BaseModel):
    email: str
    password: str
    name: Optional[str] = "Mentor"

class UserLogin(BaseModel):
    email: str
    password: str


class FeedbackInput(BaseModel):
    student_name: Optional[str] = "Anonymous"
    student_id: str
    programme: Optional[str] = ""
    assessment_session_id: Optional[str] = ""
    mentor_email: Optional[str] = ""
    improvement_status: str
    support_helpfulness: str
    follow_up_needed: bool = False
    student_feedback_summary: Optional[str] = ""
    mentor_follow_up_notes: Optional[str] = ""

    @field_validator("improvement_status")
    @classmethod
    def validate_improvement_status(cls, value):
        allowed = {"improved", "no_change", "worsened", "unsure"}
        if value not in allowed:
            raise ValueError(f"improvement_status must be one of {sorted(allowed)}")
        return value

    @field_validator("support_helpfulness")
    @classmethod
    def validate_support_helpfulness(cls, value):
        allowed = {"helpful", "partly_helpful", "not_helpful", "unknown"}
        if value not in allowed:
            raise ValueError(f"support_helpfulness must be one of {sorted(allowed)}")
        return value


class FeedbackRequestInput(BaseModel):
    student_name: Optional[str] = "Anonymous"
    student_id: str
    student_email: str
    programme: Optional[str] = ""
    assessment_session_id: Optional[str] = ""
    mentor_email: Optional[str] = ""

    @field_validator("student_email", "mentor_email")
    @classmethod
    def validate_emails(cls, value):
        value = value.strip()
        if value and "@" not in value:
            raise ValueError("A valid email address is required")
        return value


class StudentFeedbackSubmission(BaseModel):
    improvement_status: str
    support_helpfulness: str
    follow_up_needed: bool = False
    student_feedback_summary: Optional[str] = ""

    @field_validator("improvement_status")
    @classmethod
    def validate_improvement_status(cls, value):
        allowed = {"improved", "no_change", "worsened", "unsure"}
        if value not in allowed:
            raise ValueError(f"improvement_status must be one of {sorted(allowed)}")
        return value

    @field_validator("support_helpfulness")
    @classmethod
    def validate_support_helpfulness(cls, value):
        allowed = {"helpful", "partly_helpful", "not_helpful", "unknown"}
        if value not in allowed:
            raise ValueError(f"support_helpfulness must be one of {sorted(allowed)}")
        return value

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# API Routes
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/")
def health_check():
    mongo_connected = _MONGO_AVAILABLE and is_connected()
    memory_mode = _MONGO_AVAILABLE and using_memory()
    return {
        "status": "running",
        "version": "2.1.0",
        "message": "âœ… Backend is running and ready to accept requests!",
        "storage": {
            "mode": "in-memory (MongoDB auth failed â€” using fallback)" if memory_mode else (
                    "MongoDB Atlas" if mongo_connected else "unavailable"),
            "mongodb_connected": mongo_connected,
            "note": (
                "Data is stored in memory and will be lost on restart. "
                "Fix your MONGO_URI credentials to enable persistent storage."
            ) if memory_mode else None,
        },
        "components": {
            "ml_model":  model is not None,
            "xai":       explainer is not None,
            "rag":       rag_engine is not None,
            "database":  _MONGO_AVAILABLE,
        }
    }


@app.get("/api/ping")
def ping():
    """Simple connectivity test â€” always returns a message."""
    return {"message": "ðŸ“ pong â€” backend is alive!", "status": "ok"}

@app.post("/api/auth/signup")
def signup(payload: UserSignup):
    if not _MONGO_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    
    existing = get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    # Use bcrypt directly to avoid passlib versioning bugs
    hashed_pw = bcrypt.hashpw(payload.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user_id = create_user(payload.email, hashed_pw, payload.name)
    
    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to create user")
        
    return {"message": "User created successfully", "user": {"email": payload.email, "name": payload.name, "id": user_id}}

@app.post("/api/auth/login")
def login(payload: UserLogin):
    if not _MONGO_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
        
    user = get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    # Use bcrypt directly to verify
    if not bcrypt.checkpw(payload.password.encode('utf-8'), user["password_hash"].encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    return {
        "message": "Login successful", 
        "user": {
            "email": user["email"], 
            "name": user.get("name", "Mentor"),
            "id": str(user["_id"])
        }
    }


@app.get("/api/db/stats")
def db_stats():
    """MongoDB connection status and collection counts."""
    if not _MONGO_AVAILABLE:
        return {"connected": False, "reason": "pymongo not installed"}
    return get_db_stats()


@app.get("/api/history")
def recent_history(limit: int = 20):
    """Return the most recent assessment sessions from MongoDB."""
    if not _MONGO_AVAILABLE:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    return {"sessions": get_recent_assessments(limit=limit)}


@app.get("/api/history/{student_id}")
def student_history(student_id: str):
    """Return all assessment sessions for a specific student ID."""
    if not _MONGO_AVAILABLE:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    records = get_student_history(student_id)
    return {"student_id": student_id, "sessions": records, "count": len(records)}


@app.get("/api/feedback/{student_id}")
def student_feedback_history(student_id: str):
    """Return follow-up feedback records for a specific student ID."""
    if not _MONGO_AVAILABLE:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    records = get_feedback_history(student_id)
    return {"student_id": student_id, "feedback": records, "count": len(records)}


@app.post("/api/feedback/request")
def create_student_feedback_request(payload: FeedbackRequestInput):
    """Create a student-facing feedback form link that a mentor can share."""
    request_record = create_feedback_request(
        {
            "student_name": payload.student_name,
            "student_id": payload.student_id,
            "student_email": payload.student_email,
            "programme": payload.programme,
            "assessment_session_id": payload.assessment_session_id,
            "mentor_email": payload.mentor_email,
        }
    )
    link = GOOGLE_FEEDBACK_FORM_URL or f"{FRONTEND_BASE_URL}/?feedback_token={request_record['token']}"
    email_result = send_feedback_email(
        student_email=payload.student_email,
        student_name=payload.student_name or "Student",
        feedback_link=link,
        mentor_email=payload.mentor_email or "",
    )
    return {
        "student_name": payload.student_name,
        "student_id": payload.student_id,
        "student_email": payload.student_email,
        "assessment_session_id": payload.assessment_session_id,
        "token": request_record["token"],
        "feedback_link": link,
        "status": request_record["status"],
        **email_result,
    }


@app.get("/api/feedback/request/{token}")
def get_student_feedback_request(token: str):
    """Fetch feedback request metadata for a student-facing follow-up form."""
    record = get_feedback_request_by_token(token)
    if not record:
        raise HTTPException(status_code=404, detail="Feedback request not found")
    return record

@app.get("/api/config/questions")
def get_questions():
    """Return all question definitions for the frontend."""
    return {
        "screening_questions": SCREENING_QUESTIONS,
        "detailed_questions":  DETAILED_QUESTIONS,
        "domain_map":          DOMAIN_MAP,
        "likert_scale": [
            {"value": 1, "label": "Strongly Disagree"},
            {"value": 2, "label": "Disagree"},
            {"value": 3, "label": "Neutral"},
            {"value": 4, "label": "Agree"},
            {"value": 5, "label": "Strongly Agree"},
        ]
    }

@app.get("/api/models/metrics")
def get_model_metrics():
    """Return ML model performance metrics."""
    metrics_path = os.path.join(ML_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not found. Run ml/train.py first.")
    with open(metrics_path) as f:
        return json.load(f)


def send_feedback_email(
    student_email: str,
    student_name: str,
    feedback_link: str,
    mentor_email: str = "",
) -> Dict:
    """Send the student feedback form link using Gmail SMTP."""
    if not GMAIL_SENDER or not GMAIL_APP_PASSWORD:
        return {
            "email_sent": False,
            "email_error": "Project Gmail sender is not configured. Set GMAIL_SENDER_EMAIL and GMAIL_APP_PASSWORD in the backend environment.",
        }

    subject = "Student Follow-Up Feedback Form"
    mentor_line = f"\nYour mentor: {mentor_email}" if mentor_email else ""
    body = (
        f"Hello {student_name or 'Student'},\n\n"
        "Your mentor has requested a short follow-up response to understand whether the recent support was helpful "
        "and whether you feel things are improving.\n"
        "This is not a diagnosis form.\n\n"
        f"Complete the Google Form here:\n{feedback_link}\n"
        f"{mentor_line}\n\n"
        "Thank you."
    )

    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = GMAIL_SENDER
    message["To"] = student_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_SENDER, [student_email], message.as_string())
        return {"email_sent": True, "email_error": ""}
    except Exception as e:
        return {"email_sent": False, "email_error": str(e)}

@app.post("/api/screen")
def stage1_screen(payload: ScreeningInput):
    """
    Stage 1: Weekly 10-question screening.
    Returns triage result and recommendation to proceed to Stage 2.
    """
    result = screen_classify(payload.answers)

    # Domain breakdown for the 10 screening questions
    screening_domains = [q["domain"] for q in SCREENING_QUESTIONS]
    domain_totals: Dict[str, List[int]] = {}
    for idx, domain in enumerate(screening_domains):
        domain_totals.setdefault(domain, []).append(payload.answers[idx])

    domain_summary = {
        domain: {"avg": round(sum(vals) / len(vals), 2), "count": len(vals)}
        for domain, vals in domain_totals.items()
    }

    response = {
        "student_name":       payload.student_name,
        "student_id":         payload.student_id,
        "stage":              "Stage 1 - Weekly Screening",
        "risk_label":         result["risk_label"],
        "risk_class":         result["risk_class"],
        "total_score":        result["total_score"],
        "max_score":          result["max_score"],
        "proceed_to_stage2":  result["proceed_to_stage2"],
        "triage_message":     result["triage_message"],
        "domain_summary":     domain_summary,
    }

    # â”€â”€ Persist to MongoDB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if _MONGO_AVAILABLE:
        db_id = save_screening_session(
            payload={"student_name": payload.student_name, "student_id": payload.student_id, "answers": payload.answers},
            result=response,
        )
        if db_id:
            response["db_session_id"] = db_id

    return response

@app.post("/api/predict")
def stage2_predict(payload: AssessmentInput):
    """
    Stage 2: Full 25-question ML assessment with XAI and RAG.
    Returns hybrid risk classification, SHAP explanations, and mentor actions.
    """
    answers_np = np.array([payload.answers])
    answers_df = pd.DataFrame([payload.answers], columns=model_metadata.get("features", FEATURE_NAMES))

    # â”€â”€ ML Prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ml_class = 1  # default Moderate if model unavailable
    if model is not None:
        try:
            ml_class = int(model.predict(answers_df)[0])
        except Exception as e:
            print(f"[Backend] ML prediction error: {e}")

    # â”€â”€ Hybrid Classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    hybrid = hybrid_classify(payload.answers, ml_class)
    final_class = hybrid["final_class"]
    risk_label  = hybrid["final_label"]

    # â”€â”€ XAI Explanation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    xai_result = {
        "top_indicators": ["General stress indicators"],
        "feature_attributions": [],
        "domain_scores": {},
        "all_shap_values": [],
        "consistency_check": {"is_consistent": True, "warnings": []},
    }

    if explainer is not None:
        try:
            xai_result = explainer.explain(answers_np, final_class)
        except Exception as e:
            print(f"[Backend] XAI error: {e}")

    # Build domain scores from answers if XAI failed
    if not xai_result.get("domain_scores"):
        for domain, indices in DOMAIN_MAP.items():
            valid_vals = [payload.answers[i] for i in indices if i < 25]
            xai_result["domain_scores"][domain] = {
                "avg_input": round(sum(valid_vals) / len(valid_vals), 2) if valid_vals else 0,
                "shap_sum":  0.0,
            }

    # â”€â”€ RAG Mentor Guidance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    rag_result = {
        "suggested_action":       "Monitor the student and consult institutional support resources.",
        "monitoring_recommendation": "",
        "retrieved_context":      "",
        "knowledge_guidance":     "",
        "verification":           {"is_valid": True, "issues": []},
    }

    if rag_engine is not None:
        try:
            rag_result = rag_engine.generate_suggestion(
                risk_level=risk_label,
                key_indicators=xai_result["top_indicators"],
                domain_scores=xai_result["domain_scores"],
            )
        except Exception as e:
            print(f"[Backend] RAG error: {e}")

    # â”€â”€ Compose Response â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    response = {
        "student_name": payload.student_name,
        "student_id":   payload.student_id,
        "programme":    payload.programme,
        "stage":        "Stage 2 - Detailed ML Assessment",

        # Risk
        "risk_label":   risk_label,
        "risk_class":   final_class,

        # Hybrid classification breakdown
        "hybrid_decision": hybrid,

        # XAI
        "xai": {
            "top_indicators":       xai_result["top_indicators"],
            "feature_attributions": xai_result.get("feature_attributions", []),
            "domain_scores":        xai_result.get("domain_scores", {}),
            "all_shap_values":      xai_result.get("all_shap_values", []),
            "consistency_check":    xai_result.get("consistency_check", {}),
        },

        # RAG
        "rag": {
            "suggested_action":          rag_result["suggested_action"],
            "monitoring_recommendation": rag_result["monitoring_recommendation"],
            "knowledge_guidance":        rag_result.get("knowledge_guidance", ""),
            "llm_guidance":              rag_result.get("llm_guidance", ""),
            "retrieved_context":         rag_result.get("retrieved_context", ""),
            "verification":              rag_result.get("verification", {}),
        },
    }

    # â”€â”€ Persist to MongoDB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if _MONGO_AVAILABLE:
        db_id = save_assessment_session(
            payload={
                "student_name": payload.student_name,
                "student_id":   payload.student_id,
                "programme":    payload.programme,
                "answers":      payload.answers,
            },
            result=response,
        )
        if db_id:
            response["db_session_id"] = db_id

    return response


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackInput):
    """
    Save mentor follow-up feedback on whether support helped and whether the
    student appears to be improving. This does not record diagnoses.
    """
    response = {
        "student_name": payload.student_name,
        "student_id": payload.student_id,
        "programme": payload.programme,
        "assessment_session_id": payload.assessment_session_id,
        "mentor_email": payload.mentor_email,
        "improvement_status": payload.improvement_status,
        "support_helpfulness": payload.support_helpfulness,
        "follow_up_needed": payload.follow_up_needed,
        "student_feedback_summary": payload.student_feedback_summary,
        "mentor_follow_up_notes": payload.mentor_follow_up_notes,
        "stage": "Post-Intervention Feedback",
    }

    if _MONGO_AVAILABLE:
        db_id = save_feedback_session(response)
        if db_id:
            response["db_session_id"] = db_id

    return response


@app.post("/api/feedback/request/{token}/submit")
def submit_student_feedback(token: str, payload: StudentFeedbackSubmission):
    """
    Student-facing follow-up submission. Records whether the student feels the
    mentor support helped and whether they feel improved.
    """
    request_record = get_feedback_request_by_token(token)
    if not request_record:
        raise HTTPException(status_code=404, detail="Feedback request not found")
    if request_record.get("status") == "completed":
        raise HTTPException(status_code=400, detail="Feedback request already completed")

    response = {
        "student_name": request_record.get("student_name", "Anonymous"),
        "student_id": request_record.get("student_id", ""),
        "programme": request_record.get("programme", ""),
        "assessment_session_id": request_record.get("assessment_session_id", ""),
        "feedback_request_token": token,
        "mentor_email": request_record.get("mentor_email", ""),
        "improvement_status": payload.improvement_status,
        "support_helpfulness": payload.support_helpfulness,
        "follow_up_needed": payload.follow_up_needed,
        "student_feedback_summary": payload.student_feedback_summary,
        "mentor_follow_up_notes": "",
        "submitted_by": "student",
        "stage": "Post-Intervention Feedback",
    }

    db_id = save_feedback_session(response)
    if db_id:
        response["db_session_id"] = db_id
    complete_feedback_request(token)
    response["request_status"] = "completed"
    return response
