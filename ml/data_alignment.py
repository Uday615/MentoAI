import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURES = [
    "Academic coursework struggle",
    "Overwhelmed by responsibilities",
    "Requests deadline extensions",
    "Lack of consistency in assignments",
    "Sudden drop in grades",
    "Fatigue or low energy in class",
    "Decline in physical appearance",
    "Poor balance of life activities",
    "Appears stressed or irritated",
    "Anxiety or worried daily",
    "Expressions of hopelessness",
    "Low enthusiasm for learning",
    "Low motivation to achieve goals",
    "Low confidence in course subjects",
    "Observed overthinking behavior",
    "Procrastinates important tasks",
    "Nervous when speaking publicly",
    "Quiet or withdrawn with peers",
    "Reduced communication with mentor",
    "Low initiation of conversations",
    "Misses classes frequently",
    "Lack of family/friend support",
    "Spends excessive time on social media",
    "Poor behavior in group settings",
    "Low confidence overcoming challenges",
]

DOMAIN_MAP = {
    "Academic Stress": [0, 1, 2, 3, 4],
    "Sleep Disruption": [5, 6, 7],
    "Emotional Exhaustion": [8, 9, 10],
    "Motivation Decline": [11, 12, 13],
    "Cognitive Overload": [14, 15, 16],
    "Behavioral Withdrawal": [17, 18, 19, 20],
    "Support Availability": [21, 22, 23, 24],
}

DOMAIN_WEIGHTS = {
    "Academic Stress": 1.2,
    "Sleep Disruption": 1.0,
    "Emotional Exhaustion": 1.8,
    "Motivation Decline": 1.3,
    "Cognitive Overload": 1.0,
    "Behavioral Withdrawal": 1.5,
    "Support Availability": 1.1,
}

LIKERT_MAP = {
    "never": 1,
    "rarely": 2,
    "sometimes": 3,
    "often": 4,
    "always": 5,
    "not confident at all": 5,
    "slightly confident": 4,
    "moderately confident": 3,
    "very confident": 2,
    "extremely confident": 1,
    "extremely motivated": 1,
    "very motivated": 2,
    "moderately motivated": 3,
    "slightly motivated": 4,
    "not motivated at all": 5,
    "dissatisfied": 4,
    "neutral": 3,
    "satisfied": 2,
    "very satisfied": 1,
}

STUDENT_FILE = os.path.join("data", "student - Form Responses 1.csv")
MENTOR_FILE = os.path.join("data", "mentors - Form Responses 1.csv")


def normalize_scale(value, invert: bool = False) -> int:
    if pd.isna(value):
        base = 3
    else:
        text = str(value).strip().strip('"').lower()
        if text.isdigit():
            base = int(text)
        else:
            base = LIKERT_MAP.get(text, 3)
    base = max(1, min(5, int(base)))
    return 6 - base if invert else base


def _empty_feature_row() -> Dict[str, int]:
    return {feature: 2 for feature in FEATURES}


def _student_row_to_features(row: pd.Series) -> Dict[str, int]:
    record = _empty_feature_row()

    anxiety = normalize_scale(row.get("How often do you feel anxious or worried about things in your daily life?"))
    overwhelmed = normalize_scale(row.get("How often do you feel overwhelmed by responsibilities or expectations?"))
    speaking = normalize_scale(row.get("How often do you feel nervous when speaking in front of a group of people? "))
    balance = normalize_scale(row.get("How do you maintain the balance in your life?"), invert=True)
    motivation = normalize_scale(row.get("How often do you feel motivated to take steps toward achieving your personal goals?”"), invert=True)
    resilience = normalize_scale(row.get("How confident are you in overcoming challenges in life?"), invert=True)
    social_media = normalize_scale(row.get("How often do you spend time on social media?"))
    procrastination = normalize_scale(row.get("How often do you procrastinate(Postpone) important tasks?"))
    course_confidence = normalize_scale(row.get("How confident are you in understanding course subjects?"), invert=True)
    group_behavior = normalize_scale(row.get("How well you behave in a group of people(outside crowds, relatives, functions, etc)"), invert=True)
    support = normalize_scale(row.get("How often do you feel supported by your friends or family?"), invert=True)

    record["Academic coursework struggle"] = max(course_confidence, overwhelmed)
    record["Overwhelmed by responsibilities"] = overwhelmed
    record["Requests deadline extensions"] = max(1, min(5, int(round((overwhelmed + procrastination) / 2))))
    record["Lack of consistency in assignments"] = procrastination
    record["Sudden drop in grades"] = max(course_confidence, procrastination)
    record["Poor balance of life activities"] = balance
    record["Appears stressed or irritated"] = anxiety
    record["Anxiety or worried daily"] = anxiety
    record["Expressions of hopelessness"] = 5 if resilience >= 5 and anxiety >= 4 else max(1, resilience - 1)
    record["Low enthusiasm for learning"] = motivation
    record["Low motivation to achieve goals"] = motivation
    record["Low confidence in course subjects"] = course_confidence
    record["Observed overthinking behavior"] = max(anxiety, overwhelmed)
    record["Procrastinates important tasks"] = procrastination
    record["Nervous when speaking publicly"] = speaking
    record["Quiet or withdrawn with peers"] = max(group_behavior, 2)
    record["Low initiation of conversations"] = max(group_behavior, speaking)
    record["Lack of family/friend support"] = support
    record["Spends excessive time on social media"] = social_media
    record["Poor behavior in group settings"] = group_behavior
    record["Low confidence overcoming challenges"] = resilience

    return record


def _mentor_row_to_features(row: pd.Series) -> Dict[str, int]:
    record = _empty_feature_row()

    coursework = normalize_scale(row.get("Have you noticed the student struggling more than usual with coursework? "))
    withdrawn = normalize_scale(row.get("Does the student appear quieter or more withdrawn compared to before while behaving with peers?  "))
    communication = normalize_scale(row.get("Has the student reduced communication with you or avoiding your messages?  "))
    stress = normalize_scale(row.get("Does the student appear stressed, or irritated?  "))
    overthinking = normalize_scale(row.get("Have you noticed anxiety or overthinking in their behavior ? "))
    initiation = normalize_scale(row.get("  How often does the student initiate conversations, ask questions, or seek help when needed during class activities?  "), invert=True)
    balance = normalize_scale(row.get("How well does the student balance participation in different activities (sports, cultural, social)?"), invert=True)
    deadline = normalize_scale(row.get("Does the student frequently ask for deadline extensions or extra time?  "))
    consistency = normalize_scale(row.get("How consistent is the student in completing the work or assignments ? "), invert=True)
    enthusiasm = normalize_scale(row.get("How enthusiastic is the student towards learning new things?"), invert=True)
    confidence = normalize_scale(row.get("How much do you rate the student presentation in terms of confidence"), invert=True)

    record["Academic coursework struggle"] = coursework
    record["Overwhelmed by responsibilities"] = max(coursework, stress)
    record["Requests deadline extensions"] = deadline
    record["Lack of consistency in assignments"] = consistency
    record["Sudden drop in grades"] = max(coursework, consistency)
    record["Poor balance of life activities"] = balance
    record["Appears stressed or irritated"] = stress
    record["Anxiety or worried daily"] = overthinking
    record["Expressions of hopelessness"] = 5 if stress >= 5 and communication >= 4 else max(1, stress - 1)
    record["Low enthusiasm for learning"] = enthusiasm
    record["Low motivation to achieve goals"] = max(enthusiasm, initiation)
    record["Low confidence in course subjects"] = confidence
    record["Observed overthinking behavior"] = overthinking
    record["Procrastinates important tasks"] = max(consistency, 2)
    record["Nervous when speaking publicly"] = confidence
    record["Quiet or withdrawn with peers"] = withdrawn
    record["Reduced communication with mentor"] = communication
    record["Low initiation of conversations"] = initiation
    record["Misses classes frequently"] = max(withdrawn, communication)
    record["Poor behavior in group settings"] = withdrawn
    record["Low confidence overcoming challenges"] = confidence

    return record


def calculate_risk_level(row: pd.Series) -> int:
    weighted_total = 0.0
    max_possible = 0.0

    for domain, indices in DOMAIN_MAP.items():
        domain_vals = [row.iloc[i] for i in indices]
        domain_avg = np.mean(domain_vals)
        weight = DOMAIN_WEIGHTS[domain]
        weighted_total += domain_avg * weight
        max_possible += 5 * weight

    hopelessness_score = row.iloc[10]
    if hopelessness_score >= 5:
        return 2

    ratio = weighted_total / max_possible if max_possible else 0.0
    if ratio <= 0.47:
        return 0
    if ratio <= 0.68:
        return 1
    return 2


def calculate_risk_score(row: pd.Series) -> Tuple[float, float]:
    weighted_total = 0.0
    max_possible = 0.0

    for domain, indices in DOMAIN_MAP.items():
        domain_vals = [row.iloc[i] for i in indices]
        domain_avg = np.mean(domain_vals)
        weight = DOMAIN_WEIGHTS[domain]
        weighted_total += domain_avg * weight
        max_possible += 5 * weight

    ratio = weighted_total / max_possible if max_possible else 0.0
    return round(weighted_total, 4), round(ratio, 4)


def load_aligned_training_data() -> pd.DataFrame:
    student_df = pd.read_csv(STUDENT_FILE)
    mentor_df = pd.read_csv(MENTOR_FILE)

    records: List[Dict[str, object]] = []

    for _, row in student_df.iterrows():
        feature_row = _student_row_to_features(row)
        feature_series = pd.Series([feature_row[name] for name in FEATURES], index=FEATURES)
        weighted_score, ratio = calculate_risk_score(feature_series)
        records.append(
            {
                **feature_row,
                "Risk_Level": calculate_risk_level(feature_series),
                "Source": "student",
                "Weighted_Score": weighted_score,
                "Risk_Ratio": ratio,
            }
        )

    for _, row in mentor_df.iterrows():
        feature_row = _mentor_row_to_features(row)
        feature_series = pd.Series([feature_row[name] for name in FEATURES], index=FEATURES)
        weighted_score, ratio = calculate_risk_score(feature_series)
        records.append(
            {
                **feature_row,
                "Risk_Level": calculate_risk_level(feature_series),
                "Source": "mentor",
                "Weighted_Score": weighted_score,
                "Risk_Ratio": ratio,
            }
        )

    return pd.DataFrame(records)
