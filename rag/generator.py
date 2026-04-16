"""
RAG Generator Module - AI Mentor Decision Support System
Generates verified, evidence-grounded mentor action suggestions.
Uses FAISS vector DB + extractive retrieval + verification layer.
Can optionally use a local Ollama model for grounded mentor guidance.
"""
import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

EVIDENCE_KNOWLEDGE_BASE = {
    "Academic Stress": {
        "low": "Monitor academic progress periodically. Offer open-ended check-ins about coursework during scheduled sessions.",
        "moderate": "Arrange a focused academic review session. Help the student break down tasks into manageable steps and create a weekly plan. Discuss workload prioritization strategies.",
        "high": "Schedule an immediate one-on-one academic support meeting. Coordinate with faculty and academic advisors. Explore possibilities for workload adjustments or deadline extensions per institutional policy.",
    },
    "Sleep Disruption": {
        "low": "Briefly inquire about rest and self-care habits in routine meetings.",
        "moderate": "Address sleep health in your next meeting. Provide information on student wellness resources. Discuss time management for rest.",
        "high": "Refer the student to the campus health and wellness centre. Address the sleep issue as an acute concern affecting academic functioning. Follow up within 48 hours.",
    },
    "Emotional Exhaustion": {
        "low": "Check in on the student's overall wellbeing. Validate their feelings without providing therapy.",
        "moderate": "Create a safe, non-judgmental space for the student to express stress. Explore sources of exhaustion. Signpost to counselling or peer support services.",
        "high": "Urgent mentor action: connect the student with professional psychological support services immediately. Notify the appropriate duty-of-care contact per institutional protocol. Document your intervention.",
    },
    "Motivation Decline": {
        "low": "Encourage engagement and acknowledge small achievements during meetings.",
        "moderate": "Use motivational interviewing techniques to explore the student's goals and reconnect them to their academic purpose. Set short-term, achievable milestones together.",
        "high": "Initiate a structured goal-setting plan. Liaise with academic staff to explore engagement strategies. Consider referral to a study skills or academic coaching programme.",
    },
    "Cognitive Overload": {
        "low": "Check in on study methods during routine meetings. Offer resources on effective study techniques.",
        "moderate": "Discuss cognitive load and study strategies. Help the student identify and remove non-essential tasks. Recommend campus learning support resources.",
        "high": "Coordinate with the student's teaching team to assess academic demands. Consider a formal workload review. Provide access to study skills coaching and structured task-management support.",
    },
    "Behavioral Withdrawal": {
        "low": "Invite the student to group activities or peer learning sessions. Note any increasing patterns of isolation.",
        "moderate": "Proactively reach out and do not wait for the student to initiate contact. Gently address noted withdrawal and explore barriers to engagement. Encourage peer connection.",
        "high": "Activate welfare concern protocols. Escalate to the student welfare team. Arrange an urgent welfare check and document all observations thoroughly.",
    },
    "Support Availability": {
        "low": "Affirm your availability as a mentor. Remind the student of campus support services.",
        "moderate": "Explore the student's support network. Discuss barriers to seeking help. Connect them with relevant peer mentoring or student support services.",
        "high": "As the primary support figure, act urgently. Refer to specialist student support services. Ensure the student is not isolated. Coordinate with faculty and welfare contacts.",
    },
}

MONITORING_RECOMMENDATIONS = {
    "Low Risk": "Continue standard weekly screening. Re-assess in the next scheduled cycle. No immediate intervention required.",
    "Moderate Risk": "Repeat the full 25-item assessment within 1-2 weeks. Schedule a follow-up meeting after the next weekly screening to track trajectory.",
    "High Risk": "Reassess within 3-5 days after initial intervention. Maintain daily informal contact. Escalate if the student's risk does not reduce within one week.",
}

UNSAFE_PHRASES = [
    "ignore",
    "not serious",
    "wait and see",
    "do nothing",
    "will resolve itself",
    "normal behaviour",
    "overreacting",
    "therapy",
    "diagnose",
    "prescribe",
    "medication",
    "disorder",
    "clinical",
    "psychiatric",
]

REQUIRED_PHRASES_BY_RISK = {
    "High Risk": ["immediate", "urgent", "refer", "support"],
    "Moderate Risk": ["schedule", "meet", "discuss", "monitor"],
    "Low Risk": ["monitor", "check", "routine"],
}


def verify_suggestion(suggestion: str, risk_level: str) -> Dict:
    issues = []
    lower = suggestion.lower()

    found_unsafe = [phrase for phrase in UNSAFE_PHRASES if phrase in lower]
    if found_unsafe:
        issues.append(f"Removed unsafe terms: {', '.join(found_unsafe)}")
        for phrase in found_unsafe:
            suggestion = re.sub(re.escape(phrase), "[REMOVED]", suggestion, flags=re.IGNORECASE)

    required = REQUIRED_PHRASES_BY_RISK.get(risk_level, [])
    missing = [term for term in required if term not in lower]
    if missing:
        issues.append(f"Action terms expected for {risk_level}: {', '.join(missing)}")

    if len(suggestion.strip()) < 40:
        issues.append("Suggestion too short; supplementing with default guidance.")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "cleaned_suggestion": suggestion.strip(),
    }


class MentorRAG:
    def __init__(self, vector_db_path: Optional[str] = None):
        if vector_db_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            vector_db_path = os.path.join(base, "vector_db")

        self.vector_db_path = vector_db_path
        self.vectorstore = None
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "").strip()
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

        if _LANGCHAIN_AVAILABLE and os.path.exists(vector_db_path):
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={"local_files_only": True},
                )
                self.vectorstore = FAISS.load_local(
                    vector_db_path,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                print("[RAG] Vector DB loaded successfully.")
            except Exception as e:
                print(f"[RAG] Vector DB unavailable offline: {e}. Using knowledge base only.")
        elif _LANGCHAIN_AVAILABLE:
            print(f"[RAG] Vector DB not found at {vector_db_path}. Using knowledge base only.")

    def _rank_domains(self, domain_scores: Dict) -> List[Dict]:
        ranked = []
        for domain, data in domain_scores.items():
            if not isinstance(data, dict):
                continue
            avg_input = float(data.get("avg_input", 0) or 0)
            shap_sum = float(data.get("shap_sum", 0) or 0)
            severity = "high" if avg_input >= 4.2 else "moderate" if avg_input >= 3.2 else "low"
            ranking_score = avg_input + max(shap_sum, 0)
            ranked.append(
                {
                    "domain": domain,
                    "avg_input": round(avg_input, 2),
                    "shap_sum": round(shap_sum, 4),
                    "severity": severity,
                    "ranking_score": ranking_score,
                }
            )
        ranked.sort(key=lambda item: item["ranking_score"], reverse=True)
        return ranked

    def _retrieve_context(
        self,
        risk_level: str,
        indicators: List[str],
        ranked_domains: Optional[List[Dict]] = None,
        top_k: int = 3,
    ) -> str:
        if not self.vectorstore:
            return ""
        try:
            domain_text = ", ".join(item["domain"] for item in (ranked_domains or [])[:3])
            indicator_text = ", ".join(indicators[:5]) if indicators else "general distress indicators"
            query = (
                f"Mentor intervention strategies for {risk_level} student distress with domains: "
                f"{domain_text or 'general distress'} and indicators: {indicator_text}"
            )
            docs = self.vectorstore.similarity_search(query, k=top_k)
            chunks = []
            for doc in docs:
                text = re.sub(r"\s+", " ", doc.page_content.strip())
                chunks.append(text[:300])
            return "\n\n".join(chunks)
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return ""

    def _build_case_profile(
        self,
        risk_level: str,
        key_indicators: List[str],
        domain_scores: Dict,
        feature_attributions: Optional[List[Dict]] = None,
    ) -> Dict:
        ranked_domains = self._rank_domains(domain_scores)
        strongest_domains = ranked_domains[:3]
        top_features = []
        for feature in feature_attributions or []:
            if feature.get("input_value", 0) >= 4:
                top_features.append(feature.get("feature", ""))
            if len(top_features) >= 4:
                break
        if not top_features:
            top_features = key_indicators[:4]

        immediate_flags = []
        if risk_level == "High Risk":
            immediate_flags.append("Escalate same day if the student remains unreachable or visibly deteriorating.")
        if any(item["domain"] == "Emotional Exhaustion" and item["severity"] == "high" for item in strongest_domains):
            immediate_flags.append("Prioritize an emotionally safe check-in and rapid referral to approved support channels.")
        if any(item["domain"] == "Behavioral Withdrawal" and item["severity"] != "low" for item in strongest_domains):
            immediate_flags.append("Do not wait for the student to re-initiate contact; outreach should be proactive.")

        return {
            "risk_level": risk_level,
            "ranked_domains": ranked_domains,
            "strongest_domains": strongest_domains,
            "top_features": [item for item in top_features if item],
            "immediate_flags": immediate_flags,
            "key_indicators": key_indicators[:5],
        }

    def _build_case_summary(self, case_profile: Dict) -> str:
        domain_bits = [
            f"{item['domain']} ({item['severity']}, avg {item['avg_input']}/5)"
            for item in case_profile["strongest_domains"]
        ]
        indicator_bits = case_profile["top_features"] or case_profile["key_indicators"]
        parts = [
            f"Current case pattern: {case_profile['risk_level']}.",
            f"Strongest contributing domains: {', '.join(domain_bits) if domain_bits else 'no strong domain pattern identified'}.",
            f"Most relevant observed indicators: {', '.join(indicator_bits) if indicator_bits else 'general distress indicators'}.",
        ]
        if case_profile["immediate_flags"]:
            parts.append(f"Escalation flags: {' '.join(case_profile['immediate_flags'])}")
        return "\n".join(parts)

    def _build_dynamic_action_plan(self, risk_level: str, case_profile: Dict) -> str:
        level_key = risk_level.lower().replace(" risk", "")
        strongest_domains = case_profile["strongest_domains"]
        domain_actions = []
        for item in strongest_domains:
            domain = item["domain"]
            domain_guidance = EVIDENCE_KNOWLEDGE_BASE.get(domain, {}).get(level_key, "")
            if not domain_guidance:
                continue
            modifier = {
                "high": "Treat this as a primary focus in the next mentor contact.",
                "moderate": "Address this directly in the next scheduled mentor meeting.",
                "low": "Keep this under routine observation while reinforcing protective factors.",
            }.get(item["severity"], "")
            domain_actions.append(f"[{domain}] {domain_guidance} {modifier}".strip())

        priority_features = case_profile["top_features"][:3]
        if priority_features:
            feature_line = (
                "Focus the conversation on these concrete observations: "
                + "; ".join(priority_features)
                + "."
            )
        else:
            feature_line = "Focus the conversation on the student’s current academic, emotional, and support barriers."

        coordination_line = {
            "High Risk": "Coordinate quickly with approved student support or welfare pathways and document the mentor response.",
            "Moderate Risk": "Agree on one short-term support plan, one check-in date, and one escalation trigger if the student worsens.",
            "Low Risk": "Use a light-touch support plan with one clear follow-up check and reinforce existing support resources.",
        }.get(risk_level, "Maintain regular mentor contact and review the case again in the next cycle.")

        if not domain_actions:
            domain_actions.append(coordination_line)

        return "\n\n".join(
            [
                self._build_case_summary(case_profile),
                "Recommended mentor actions:",
                "\n".join(f"- {item}" for item in domain_actions),
                feature_line,
                coordination_line,
            ]
        )

    def _generate_with_ollama(
        self,
        risk_level: str,
        key_indicators: List[str],
        knowledge_guidance: str,
        retrieved_context: str,
        case_summary: str,
    ) -> str:
        if not self.ollama_model:
            return ""

        prompt = (
            "You are supporting a university mentor. "
            "Do not diagnose, do not provide therapy, and do not classify risk. "
            "Use only the supplied risk level, indicators, and context to write 2 short mentor-action paragraphs.\n\n"
            f"Risk Level: {risk_level}\n"
            f"Indicators: {', '.join(key_indicators) if key_indicators else 'General distress indicators'}\n\n"
            f"Case Summary:\n{case_summary}\n\n"
            f"Grounded Guidance:\n{knowledge_guidance}\n\n"
            f"Retrieved Context:\n{retrieved_context or 'No retrieved context available.'}\n\n"
            "Return mentor-facing actions only."
        )

        payload = json.dumps(
            {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body.get("response", "").strip()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"[RAG] Ollama generation unavailable: {e}")
            return ""

    def generate_suggestion(
        self,
        risk_level: str,
        key_indicators: List[str],
        domain_scores: Optional[Dict] = None,
        feature_attributions: Optional[List[Dict]] = None,
    ) -> Dict:
        if domain_scores is None:
            domain_scores = {}

        case_profile = self._build_case_profile(
            risk_level=risk_level,
            key_indicators=key_indicators,
            domain_scores=domain_scores,
            feature_attributions=feature_attributions,
        )
        case_summary = self._build_case_summary(case_profile)
        knowledge_guidance = self._build_dynamic_action_plan(risk_level, case_profile)
        retrieved_context = self._retrieve_context(
            risk_level,
            key_indicators,
            ranked_domains=case_profile["ranked_domains"],
        )
        ollama_guidance = self._generate_with_ollama(
            risk_level=risk_level,
            key_indicators=key_indicators,
            knowledge_guidance=knowledge_guidance,
            retrieved_context=retrieved_context,
            case_summary=case_summary,
        )

        suggestion_parts = []
        if ollama_guidance:
            suggestion_parts.append(ollama_guidance)
        else:
            suggestion_parts.append(knowledge_guidance)

        suggestion_parts.append(
            f"Monitoring Recommendation:\n{MONITORING_RECOMMENDATIONS.get(risk_level, 'Repeat screening at next cycle.')}"
        )

        if retrieved_context:
            suggestion_parts.append(f"Retrieved Context:\n{retrieved_context[:400]}")

        full_suggestion = "\n\n".join(part for part in suggestion_parts if part)
        verification = verify_suggestion(full_suggestion, risk_level)

        return {
            "suggested_action": verification["cleaned_suggestion"],
            "monitoring_recommendation": MONITORING_RECOMMENDATIONS.get(risk_level, ""),
            "retrieved_context": retrieved_context[:300] if retrieved_context else "",
            "case_summary": case_summary,
            "knowledge_guidance": knowledge_guidance,
            "llm_guidance": ollama_guidance,
            "verification": verification,
        }
