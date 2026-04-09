import { useState, useEffect, useRef } from 'react'
import { SignedIn, SignedOut, SignIn, SignUp, UserButton, useUser } from '@clerk/clerk-react'
import './index.css'

// ─── Constants ───────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000'

const LIKERT_LABELS = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
const IMPROVEMENT_OPTIONS = [
  { value: 'improved', label: 'Improved' },
  { value: 'no_change', label: 'No major change' },
  { value: 'worsened', label: 'Worsened' },
  { value: 'unsure', label: 'Unsure' },
]

const HELPFULNESS_OPTIONS = [
  { value: 'helpful', label: 'Helpful' },
  { value: 'partly_helpful', label: 'Partly helpful' },
  { value: 'not_helpful', label: 'Not helpful' },
  { value: 'unknown', label: 'Unknown' },
]

const DOMAIN_ICONS = {
  'Academic Stress':       '📚',
  'Sleep Disruption':      '😴',
  'Emotional Exhaustion':  '🔥',
  'Motivation Decline':    '📉',
  'Cognitive Overload':    '🧠',
  'Behavioral Withdrawal': '🚪',
  'Support Availability':  '🤝',
}

const CLERK_APPEARANCE = {
  variables: {
    colorPrimary: '#6366f1',
    colorBackground: 'rgba(15, 23, 42, 0.92)',
    colorInputBackground: '#0b1220',
    colorInputText: '#f8fafc',
    colorText: '#e5e7eb',
    colorTextSecondary: '#94a3b8',
    colorDanger: '#f87171',
    colorSuccess: '#34d399',
    borderRadius: '14px',
  },
  elements: {
    card: {
      background: 'transparent',
      boxShadow: 'none',
      border: 'none',
      width: '100%',
    },
    rootBox: {
      width: '100%',
    },
    headerTitle: {
      color: '#f8fafc',
    },
    headerSubtitle: {
      color: '#94a3b8',
    },
    formFieldLabel: {
      color: '#cbd5e1',
      fontWeight: '600',
    },
    formFieldInput: {
      background: '#0b1220',
      color: '#f8fafc',
      border: '1px solid rgba(148, 163, 184, 0.28)',
      boxShadow: 'none',
    },
    formButtonPrimary: {
      background: 'linear-gradient(135deg, #6366f1, #4f46e5)',
      color: '#ffffff',
      boxShadow: '0 10px 24px rgba(99, 102, 241, 0.35)',
    },
    footerActionText: {
      color: '#94a3b8',
    },
    footerActionLink: {
      color: '#a5b4fc',
    },
    identityPreviewText: {
      color: '#e5e7eb',
    },
    socialButtonsBlockButton: {
      background: '#111827',
      border: '1px solid rgba(148, 163, 184, 0.24)',
      color: '#f8fafc',
    },
    formResendCodeLink: {
      color: '#a5b4fc',
    },
    otpCodeFieldInput: {
      background: '#0b1220',
      color: '#f8fafc',
      border: '1px solid rgba(148, 163, 184, 0.28)',
    },
    alertText: {
      color: '#fecaca',
    },
    formFieldSuccessText: {
      color: '#86efac',
    },
  },
}

const SCREENING_QUESTIONS = [
  { id: 'sq1',  text: 'The student appears stressed or emotionally irritated.',          domain: 'Emotional Exhaustion' },
  { id: 'sq2',  text: 'The student shows signs of low motivation or disengagement.',     domain: 'Motivation Decline' },
  { id: 'sq3',  text: 'The student seems withdrawn or avoids peer interaction.',         domain: 'Behavioral Withdrawal' },
  { id: 'sq4',  text: 'The student has missed recent classes or scheduled sessions.',    domain: 'Behavioral Withdrawal' },
  { id: 'sq5',  text: 'The student appears fatigued or has low energy consistently.',    domain: 'Sleep Disruption' },
  { id: 'sq6',  text: 'The student has expressed worry, anxiety, or hopelessness.',      domain: 'Emotional Exhaustion' },
  { id: 'sq7',  text: 'The student is struggling with academic coursework or deadlines.', domain: 'Academic Stress' },
  { id: 'sq8',  text: 'The student has reduced communication with me as their mentor.',  domain: 'Behavioral Withdrawal' },
  { id: 'sq9',  text: 'The student seems to lack confidence in their academic ability.', domain: 'Motivation Decline' },
  { id: 'sq10', text: 'The student mentions lacking support from family or friends.',    domain: 'Support Availability' },
]

const DETAILED_QUESTIONS = [
  // Academic Stress
  { id: 'q1',  text: 'The student struggles significantly with academic coursework.',       domain: 'Academic Stress' },
  { id: 'q2',  text: 'The student appears overwhelmed by responsibilities.',                domain: 'Academic Stress' },
  { id: 'q3',  text: 'The student frequently requests extensions for deadlines.',           domain: 'Academic Stress' },
  { id: 'q4',  text: 'The student shows inconsistency in assignment submission.',           domain: 'Academic Stress' },
  { id: 'q5',  text: 'There has been a noticeable drop in the student\'s grades.',         domain: 'Academic Stress' },
  // Sleep Disruption
  { id: 'q6',  text: 'The student consistently shows fatigue or low energy in sessions.',  domain: 'Sleep Disruption' },
  { id: 'q7',  text: 'The student shows a visible decline in physical self-presentation.', domain: 'Sleep Disruption' },
  { id: 'q8',  text: 'The student reports or shows poor balance across life activities.',  domain: 'Sleep Disruption' },
  // Emotional Exhaustion
  { id: 'q9',  text: 'The student frequently appears stressed, tense, or irritated.',      domain: 'Emotional Exhaustion' },
  { id: 'q10', text: 'The student expresses anxiety or worry on a daily basis.',           domain: 'Emotional Exhaustion' },
  { id: 'q11', text: 'The student has expressed feelings of hopelessness or despair.',     domain: 'Emotional Exhaustion' },
  // Motivation Decline
  { id: 'q12', text: 'The student shows low enthusiasm for learning activities.',          domain: 'Motivation Decline' },
  { id: 'q13', text: 'The student demonstrates low motivation to achieve their goals.',    domain: 'Motivation Decline' },
  { id: 'q14', text: 'The student lacks confidence in their academic subject areas.',      domain: 'Motivation Decline' },
  // Cognitive Overload
  { id: 'q15', text: 'The student shows signs of overthinking or cognitive rumination.',   domain: 'Cognitive Overload' },
  { id: 'q16', text: 'The student procrastinates on important academic tasks.',            domain: 'Cognitive Overload' },
  { id: 'q17', text: 'The student is nervous or anxious when speaking publicly.',         domain: 'Cognitive Overload' },
  // Behavioral Withdrawal
  { id: 'q18', text: 'The student is quiet or withdrawn in peer settings.',               domain: 'Behavioral Withdrawal' },
  { id: 'q19', text: 'The student has reduced communication with their mentor.',          domain: 'Behavioral Withdrawal' },
  { id: 'q20', text: 'The student rarely initiates conversations or interactions.',       domain: 'Behavioral Withdrawal' },
  { id: 'q21', text: 'The student misses classes or scheduled sessions frequently.',      domain: 'Behavioral Withdrawal' },
  // Support Availability
  { id: 'q22', text: 'The student lacks visible support from family or close friends.',   domain: 'Support Availability' },
  { id: 'q23', text: 'The student spends excessive time on social media/digital devices.', domain: 'Support Availability' },
  { id: 'q24', text: 'The student displays poor behaviour in group or social settings.',  domain: 'Support Availability' },
  { id: 'q25', text: 'The student shows low confidence in overcoming personal challenges.', domain: 'Support Availability' },
]

// ─── Risk Utilities ───────────────────────────────────────────────
function getRiskClass(label) {
  if (!label) return 'low'
  const l = label.toLowerCase()
  if (l.includes('high'))     return 'high'
  if (l.includes('moderate')) return 'moderate'
  return 'low'
}

function getRiskEmoji(label) {
  const cl = getRiskClass(label)
  if (cl === 'high')     return '🔴'
  if (cl === 'moderate') return '🟡'
  return '🟢'
}

// ─── Components ───────────────────────────────────────────────────

function Header({ currentUser }) {
  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="header-icon">🎓</div>
        <div>
          <div className="header-title">MentorAI Decision Support</div>
          <div className="header-subtitle">Early Student Distress Identification System</div>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <div className="header-badge">MENTOR-FACING · v2.0</div>
        {currentUser && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              👤 {currentUser.name || currentUser.email}
            </span>
            <UserButton afterSignOutUrl="/" />
          </div>
        )}
      </div>
    </header>
  )
}


function ProgressSteps({ currentStep }) {
  const steps = [
    { label: 'Student Info',  num: 0 },
    { label: 'Stage 1 Screen', num: 1 },
    { label: 'Stage 2 Detail', num: 2 },
    { label: 'Results',        num: 3 },
  ]
  return (
    <div className="progress-steps">
      {steps.map((s, i) => (
        <>
          <div key={s.num} className="step-item">
            <div className={`step-circle ${
              currentStep === s.num ? 'active' :
              currentStep > s.num  ? 'completed' : 'pending'
            }`}>
              {currentStep > s.num ? '✓' : s.num + 1}
            </div>
            <span className="step-label">{s.label}</span>
          </div>
          {i < steps.length - 1 && (
            <div key={`conn-${i}`} className={`step-connector ${
              currentStep > i ? 'completed' : currentStep === i ? 'active' : ''
            }`} />
          )}
        </>
      ))}
    </div>
  )
}

function LikertQuestion({ question, index, value, onChange, globalIndex }) {
  return (
    <div className="question-card">
      <p className="question-text">
        <span className="question-num">Q{globalIndex + 1}.</span>
        {question.text}
      </p>
      <div className="likert-scale">
        {[1, 2, 3, 4, 5].map((val) => (
          <label key={val} className="likert-option">
            <input
              type="radio"
              name={`q-${question.id}`}
              value={val}
              checked={value === val}
              onChange={() => onChange(index, val)}
            />
            <span className="likert-btn">{val}</span>
          </label>
        ))}
      </div>
      <div className="likert-labels">
        <span className="likert-label-text">Strongly Disagree</span>
        <span className="likert-label-text" style={{textAlign:'right'}}>Strongly Agree</span>
      </div>
    </div>
  )
}

function DomainScoreBars({ domainScores, riskClass }) {
  // Build bar data from domain_scores
  const entries = Object.entries(domainScores)
    .map(([name, data]) => ({
      name,
      icon: DOMAIN_ICONS[name] || '📊',
      avg: typeof data === 'object' ? (data.avg_input || 0) : 0,
    }))
    .sort((a, b) => b.avg - a.avg)

  return (
    <div className="domain-scores">
      {entries.map(({ name, icon, avg }) => {
        const pct = ((avg / 5) * 100).toFixed(0)
        const barClass = avg >= 3.8 ? 'bar-high' : avg >= 2.6 ? 'bar-moderate' : 'bar-low'
        return (
          <div key={name} className="domain-score-item">
            <div className="domain-score-header">
              <span className="domain-score-name">{icon} {name}</span>
              <span className="domain-score-value">{avg.toFixed(1)} / 5</span>
            </div>
            <div className="score-bar-track">
              <div
                className={`score-bar-fill ${barClass}`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

function LoadingView({ steps, activeStep }) {
  return (
    <div className="loading-overlay">
      <div className="spinner" />
      <div className="loading-text">Analysing assessment data…</div>
      <div className="loading-steps">
        {steps.map((s, i) => (
          <div key={i} className={`loading-step ${i < activeStep ? 'done' : i === activeStep ? 'active' : ''}`}>
            {i < activeStep ? '✅' : i === activeStep ? '⏳' : '⬜'} {s}
          </div>
        ))}
      </div>
    </div>
  )
}

function FollowUpFeedbackCard({
  currentUser,
  fullResult,
  studentName,
  studentId,
  studentEmail,
  programme,
  generatedFeedbackLink,
  onGenerateLink,
  feedbackLinkSaving,
  emailDeliveryStatus,
}) {
  return (
    <div className="result-card full-width">
      <div className="result-card-header">
        <div className="result-card-icon">📝</div>
        <div>
          <div className="result-card-title">Student Follow-Up Form</div>
          <div className="result-card-subtitle">Send the student your Google Form by email so they can report whether support helped and whether they improved</div>
        </div>
      </div>

      <div className="feedback-meta">
        <div>Student: {studentName || 'Anonymous'} {studentId ? `· ${studentId}` : ''} {programme ? `· ${programme}` : ''}</div>
        <div>Student Email: {studentEmail || 'Not provided'}</div>
        <div>Mentor: {currentUser?.email || 'Not available'}</div>
        <div>Assessment Ref: {fullResult?.db_session_id || 'Temporary in-memory session'}</div>
      </div>

      {generatedFeedbackLink && (
        <div className="success-box">
          Google Form link ready.
          {emailDeliveryStatus && <div style={{ marginTop: '0.5rem' }}>{emailDeliveryStatus}</div>}
          <div className="feedback-link-box">{generatedFeedbackLink}</div>
        </div>
      )}

      <div className="action-row" style={{ marginTop: '1rem', justifyContent: 'flex-start' }}>
        <button className="btn btn-primary btn-sm" onClick={onGenerateLink} disabled={feedbackLinkSaving}>
          {feedbackLinkSaving ? 'Sending...' : 'Send Student Feedback Email'}
        </button>
      </div>
      <div style={{ marginTop: '0.85rem', color: 'var(--text-muted)', fontSize: '0.84rem' }}>
        Uses the project Gmail sender configured on the backend and sends your shared Google Form link.
      </div>
    </div>
  )
}

function StudentFeedbackView({ token }) {
  const [requestInfo, setRequestInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [submitted, setSubmitted] = useState(false)
  const [form, setForm] = useState({
    improvement_status: 'unsure',
    support_helpfulness: 'unknown',
    follow_up_needed: false,
    student_feedback_summary: '',
  })

  useEffect(() => {
    let ignore = false
    const loadRequest = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/feedback/request/${token}`)
        if (!res.ok) throw new Error('Unable to load feedback form.')
        const data = await res.json()
        if (!ignore) setRequestInfo(data)
      } catch (err) {
        if (!ignore) setError(err.message)
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    loadRequest()
    return () => { ignore = true }
  }, [token])

  const updateField = (field, value) => {
    setForm(prev => ({ ...prev, [field]: value }))
  }

  const submitStudentFeedback = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/feedback/request/${token}/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || 'Unable to submit feedback.')
      }
      setSubmitted(true)
      setError(null)
    } catch (err) {
      setError(err.message)
    }
  }

  if (loading) {
    return <div className="auth-wrapper"><div className="auth-card glass-panel">Loading feedback form...</div></div>
  }

  if (error) {
    return <div className="auth-wrapper"><div className="auth-card glass-panel"><div className="error-box">⚠️ {error}</div></div></div>
  }

  if (submitted) {
    return (
      <div className="auth-wrapper">
        <div className="auth-card glass-panel">
          <div className="section-header">
            <div className="stage-tag">Follow-Up Submitted</div>
            <h2 className="section-title">Thank you</h2>
            <p className="section-desc">Your feedback has been shared with your mentor to help them understand whether the support was useful and whether you are improving.</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="auth-wrapper">
      <div className="auth-card glass-panel" style={{ maxWidth: '720px' }}>
        <div className="section-header">
          <div className="stage-tag">Student Follow-Up</div>
          <h2 className="section-title">Support Feedback Form</h2>
          <p className="section-desc">
            This form asks whether the mentor support was helpful and whether you feel things are improving.
            It does not diagnose any condition.
          </p>
        </div>

        <div className="feedback-meta" style={{ marginBottom: '1rem' }}>
          <div>Student: {requestInfo?.student_name || 'Anonymous'} {requestInfo?.student_id ? `· ${requestInfo.student_id}` : ''}</div>
          <div>Programme: {requestInfo?.programme || 'Not provided'}</div>
        </div>

        <div className="feedback-grid">
          <div className="form-group">
            <label className="form-label">Do you feel improved?</label>
            <select className="form-input" value={form.improvement_status} onChange={e => updateField('improvement_status', e.target.value)}>
              {IMPROVEMENT_OPTIONS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Was the mentor support helpful?</label>
            <select className="form-input" value={form.support_helpfulness} onChange={e => updateField('support_helpfulness', e.target.value)}>
              {HELPFULNESS_OPTIONS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
            </select>
          </div>
        </div>

        <div className="form-group" style={{ marginTop: '1rem' }}>
          <label className="form-label">Anything you want your mentor to know?</label>
          <textarea
            className="form-input feedback-textarea"
            value={form.student_feedback_summary}
            onChange={e => updateField('student_feedback_summary', e.target.value)}
            placeholder="Share whether the support helped, what improved, or what follow-up would still be useful."
          />
        </div>

        <label className="feedback-checkbox">
          <input type="checkbox" checked={form.follow_up_needed} onChange={e => updateField('follow_up_needed', e.target.checked)} />
          <span>I would still like more follow-up support.</span>
        </label>

        <div className="action-row" style={{ justifyContent: 'flex-start', marginTop: '1.25rem' }}>
          <button className="btn btn-primary" onClick={submitStudentFeedback}>Submit Feedback</button>
        </div>
      </div>
    </div>
  )
}

function AuthScreen() {
  const [mode, setMode] = useState('sign-in')

  return (
    <div className="auth-wrapper">
      <div className="auth-card glass-panel" style={{animation: 'fadeSlideUp 0.6s ease-out'}}>
        <div className="auth-header" style={{textAlign: 'center', marginBottom: '2rem'}}>
          <div className="header-icon" style={{fontSize: '3rem', marginBottom: '1rem', display: 'inline-block'}}>🎓</div>
          <h2 style={{color: 'var(--text-light)', fontSize: '1.5rem', marginBottom: '0.5rem'}}>
            {mode === 'sign-in' ? 'Welcome Back' : 'Create Account'}
          </h2>
          <p style={{color: 'var(--text-secondary)', fontSize: '0.9rem'}}>
            {mode === 'sign-in' ? 'Sign in with Clerk to access your mentor dashboard' : 'Create your mentor account with Clerk'}
          </p>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center' }}>
          {mode === 'sign-in' ? (
            <SignIn
              routing="virtual"
              appearance={CLERK_APPEARANCE}
            />
          ) : (
            <SignUp
              routing="virtual"
              appearance={CLERK_APPEARANCE}
            />
          )}
        </div>

        <div style={{
          marginTop: '1rem',
          padding: '0.85rem 1rem',
          borderRadius: 'var(--radius-md)',
          border: '1px solid rgba(99,102,241,0.2)',
          background: 'rgba(99,102,241,0.06)',
          color: 'var(--text-secondary)',
          fontSize: '0.85rem',
          lineHeight: 1.6,
        }}>
          Clerk handles mentor password rules and secure authentication. In the Clerk dashboard, keep strong password rules enabled and optionally add Google sign-in if mentors should log in with Gmail directly.
        </div>

        <div className="auth-footer" style={{textAlign: 'center', marginTop: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '1.5rem'}}>
          <p style={{color: 'var(--text-secondary)', fontSize: '0.9rem'}}>
            {mode === 'sign-in' ? "Don't have an account?" : "Already have an account?"}{' '}
            <button type="button" className="auth-toggle-btn" onClick={() => setMode(mode === 'sign-in' ? 'sign-up' : 'sign-in')} style={{
              background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontWeight: '500', padding: 0
            }}>
              {mode === 'sign-in' ? 'Sign Up' : 'Log In'}
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}

// ─── Main App ──────────────────────────────────────────────────────
export default function App() {
  const { user, isSignedIn } = useUser()
  const feedbackToken = new URLSearchParams(window.location.search).get('feedback_token')
  const [step, setStep]             = useState(0)   // 0=info, 1=screen, 2=detail, 3=results
  const [studentName, setStudentName] = useState('')
  const [studentId, setStudentId]   = useState('')
  const [studentEmail, setStudentEmail] = useState('')
  const [programme, setProgramme]   = useState('')

  const [s1Answers, setS1Answers]   = useState(Array(10).fill(3))
  const [s2Answers, setS2Answers]   = useState(Array(25).fill(3))

  const [screenResult, setScreenResult] = useState(null)
  const [fullResult, setFullResult]     = useState(null)
  const [loading, setLoading]           = useState(false)
  const [loadingStep, setLoadingStep]   = useState(0)
  const [error, setError]               = useState(null)
  const [feedbackSaving, setFeedbackSaving] = useState(false)
  const [feedbackSaved, setFeedbackSaved] = useState(false)
  const [feedbackLinkSaving, setFeedbackLinkSaving] = useState(false)
  const [generatedFeedbackLink, setGeneratedFeedbackLink] = useState('')
  const [emailDeliveryStatus, setEmailDeliveryStatus] = useState('')
  const [feedbackForm, setFeedbackForm] = useState({
    improvement_status: 'unsure',
    support_helpfulness: 'unknown',
    follow_up_needed: false,
    student_feedback_summary: '',
    mentor_follow_up_notes: '',
  })

  const topRef = useRef(null)
  const currentUser = isSignedIn ? {
    email: user?.primaryEmailAddress?.emailAddress || '',
    name: user?.fullName || user?.firstName || 'Mentor',
  } : null

  const scrollTop = () => topRef.current?.scrollIntoView({ behavior: 'smooth' })

  const handleS1Change = (idx, val) => {
    const a = [...s1Answers]; a[idx] = val; setS1Answers(a)
  }

  const handleS2Change = (idx, val) => {
    const a = [...s2Answers]; a[idx] = val; setS2Answers(a)
  }

  // Submit Stage 1
  const submitStage1 = async () => {
    setLoading(true); setError(null)
    try {
      const res = await fetch(`${API_BASE}/api/screen`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_name: studentName || 'Anonymous',
          student_id:   studentId,
          answers:      s1Answers,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setScreenResult(data)
      setStep(2)  // Always show stage 2 option — mentor decides
      scrollTop()
    } catch (e) {
      setError(`Stage 1 failed: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Submit Stage 2
  const LOADING_STEPS = [
    'Applying rule-based classification…',
    'Running ML model inference…',
    'Computing SHAP explanations (XAI)…',
    'Retrieving research grounding (RAG)…',
    'Verifying suggestion safety…',
  ]

  const submitStage2 = async () => {
    setLoading(true); setError(null); setLoadingStep(0)

    // Simulate progress steps
    const stepInterval = setInterval(() => {
      setLoadingStep(prev => (prev < LOADING_STEPS.length - 1 ? prev + 1 : prev))
    }, 900)

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_name: studentName || 'Anonymous',
          student_id:   studentId,
          programme:    programme,
          answers:      s2Answers,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setFullResult(data)
      setStep(3)
      scrollTop()
    } catch (e) {
      setError(`Stage 2 analysis failed: ${e.message}`)
    } finally {
      clearInterval(stepInterval)
      setLoading(false)
    }
  }

  const resetAll = () => {
    setStep(0)
    setStudentName(''); setStudentId(''); setStudentEmail(''); setProgramme('')
    setS1Answers(Array(10).fill(3))
    setS2Answers(Array(25).fill(3))
    setScreenResult(null); setFullResult(null); setError(null)
    setFeedbackSaving(false); setFeedbackSaved(false)
    setFeedbackLinkSaving(false); setGeneratedFeedbackLink(''); setEmailDeliveryStatus('')
    setFeedbackForm({
      improvement_status: 'unsure',
      support_helpfulness: 'unknown',
      follow_up_needed: false,
      student_feedback_summary: '',
      mentor_follow_up_notes: '',
    })
    scrollTop()
  }

  const updateFeedbackField = (field, value) => {
    setFeedbackSaved(false)
    setFeedbackForm(prev => ({ ...prev, [field]: value }))
  }

  const submitFeedback = async () => {
    if (!studentId) {
      setError('Add a Student ID before saving follow-up feedback so it can be linked to the correct student.')
      return
    }

    setFeedbackSaving(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_name: studentName || 'Anonymous',
          student_id: studentId,
          programme: programme,
          assessment_session_id: fullResult?.db_session_id || '',
          mentor_email: currentUser?.email || '',
          ...feedbackForm,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      await res.json()
      setFeedbackSaved(true)
    } catch (e) {
      setError(`Follow-up feedback failed: ${e.message}`)
    } finally {
      setFeedbackSaving(false)
    }
  }

  const generateStudentFeedbackLink = async () => {
    if (!studentId) {
      setError('Add a Student ID before generating a student feedback form link.')
      return
    }
    if (!studentEmail) {
      setError('Add the student email before sending the follow-up form.')
      return
    }
    if (!currentUser?.email) {
      setError('Sign in with Clerk before sending student feedback emails.')
      return
    }

    setFeedbackLinkSaving(true)
    setError(null)
    setEmailDeliveryStatus('')
    try {
      const res = await fetch(`${API_BASE}/api/feedback/request`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          student_name: studentName || 'Anonymous',
          student_id: studentId,
          student_email: studentEmail,
          programme: programme,
          assessment_session_id: fullResult?.db_session_id || '',
          mentor_email: currentUser?.email || '',
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setGeneratedFeedbackLink(data.feedback_link)
      setEmailDeliveryStatus(
        data.email_sent
          ? `Email sent to ${studentEmail}.`
          : `Email not sent automatically: ${data.email_error || 'Unknown error'}. You can still copy the link below.`
      )
    } catch (e) {
      setError(`Feedback link generation failed: ${e.message}`)
    } finally {
      setFeedbackLinkSaving(false)
    }
  }

  // Group questions by domain
  const groupByDomain = (questions) => {
    const groups = {}
    questions.forEach((q, i) => {
      if (!groups[q.domain]) groups[q.domain] = []
      groups[q.domain].push({ ...q, localIdx: i })
    })
    return groups
  }

  const s1Groups = groupByDomain(SCREENING_QUESTIONS)
  const s2Groups = groupByDomain(DETAILED_QUESTIONS)
  if (feedbackToken) {
    return <StudentFeedbackView token={feedbackToken} />
  }

  // ── RENDER ─────────────────────────────────────────────────────
  return (
    <div className="app-wrapper" ref={topRef}>
      <SignedOut>
        <AuthScreen />
      </SignedOut>
      <SignedIn>
        <>
          <Header currentUser={currentUser} />
          <ProgressSteps currentStep={step} />

      {error && (
        <div className="error-box" style={{marginBottom: '1.5rem'}}>
          ⚠️ {error}
          <br /><small>Ensure the backend is running: <code>uvicorn main:app --reload --port 8000</code></small>
        </div>
      )}

      {/* ── STEP 0: Student Info ─────────────────────────────── */}
      {step === 0 && (
        <div className="glass-panel" style={{animation: 'fadeSlideUp 0.6s ease-out'}}>
          <div className="section-header">
            <div className="stage-tag">📋 Start Here</div>
            <h2 className="section-title">Student Information</h2>
            <p className="section-desc">
              Enter the student's details before beginning the distress screening process.
              All data is used for decision support only and is never shared externally.
            </p>
          </div>

          <div className="form-section">
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Student Name</label>
                <input
                  id="student-name"
                  className="form-input"
                  type="text"
                  placeholder="e.g. Jane Doe"
                  value={studentName}
                  onChange={e => setStudentName(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Student ID</label>
                <input
                  id="student-id"
                  className="form-input"
                  type="text"
                  placeholder="e.g. STU-2024-001"
                  value={studentId}
                  onChange={e => setStudentId(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Student Email</label>
                <input
                  id="student-email"
                  className="form-input"
                  type="email"
                  placeholder="e.g. student@college.edu"
                  value={studentEmail}
                  onChange={e => setStudentEmail(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Programme / Course</label>
                <input
                  id="student-programme"
                  className="form-input"
                  type="text"
                  placeholder="e.g. BSc Computer Science"
                  value={programme}
                  onChange={e => setProgramme(e.target.value)}
                />
              </div>
            </div>
          </div>

          <div style={{
            background: 'rgba(99,102,241,0.05)',
            border: '1px solid rgba(99,102,241,0.15)',
            borderRadius: 'var(--radius-md)',
            padding: '1rem 1.2rem',
            marginBottom: '2rem',
            fontSize: '0.85rem',
            color: 'var(--text-secondary)',
            lineHeight: 1.6,
          }}>
            <strong style={{color: 'var(--accent-light)'}}>⚠️ System Boundaries:</strong>{' '}
            This system assists mentor decision-making only. It does <strong>not</strong> diagnose
            mental health conditions, replace clinical judgment, or provide therapy.
            All suggested actions are evidence-based mentor interventions.
          </div>

          <button id="btn-start-screening" className="btn btn-primary btn-full" onClick={() => { setStep(1); scrollTop() }}>
            Begin Stage 1 — Weekly Screening →
          </button>
        </div>
      )}

      {/* ── STEP 1: Stage 1 Screening ────────────────────────── */}
      {step === 1 && !loading && (
        <div style={{animation: 'fadeSlideUp 0.6s ease-out'}}>
          <div className="glass-panel" style={{marginBottom: '2rem'}}>
            <div className="section-header">
              <div className="stage-tag">🔍 Stage 1</div>
              <h2 className="section-title">Weekly Screening Assessment</h2>
              <p className="section-desc">
                Rate each indicator based on your recent observations of the student.
                Use the Likert scale: <strong>1 = Strongly Disagree</strong> to <strong>5 = Strongly Agree</strong>.
                This 10-item triage determines whether a detailed assessment is needed.
              </p>
            </div>

            <div className="questions-wrapper">
              {Object.entries(s1Groups).map(([domain, qs]) => (
                <div key={domain} className="domain-group">
                  <div className="domain-header">
                    <span className="domain-icon">{DOMAIN_ICONS[domain] || '📊'}</span>
                    <span className="domain-title">{domain}</span>
                    <span className="domain-count">{qs.length} question{qs.length > 1 ? 's' : ''}</span>
                  </div>
                  <div className="questions-list">
                    {qs.map((q) => (
                      <LikertQuestion
                        key={q.id}
                        question={q}
                        index={q.localIdx}
                        value={s1Answers[q.localIdx]}
                        onChange={handleS1Change}
                        globalIndex={q.localIdx}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="action-row">
            <button className="btn btn-secondary" onClick={() => setStep(0)}>← Back</button>
            <button id="btn-submit-stage1" className="btn btn-primary" onClick={submitStage1}>
              Submit Screening →
            </button>
          </div>
        </div>
      )}

      {/* Loading overlay */}
      {loading && step === 1 && (
        <div className="glass-panel">
          <LoadingView steps={['Applying screening rules…', 'Scoring domains…']} activeStep={0} />
        </div>
      )}

      {/* ── STEP 2: Screening Results + Stage 2 Choice ─────── */}
      {step === 2 && !loading && (
        <div style={{animation: 'fadeSlideUp 0.6s ease-out'}}>
          {screenResult && (
            <div className="glass-panel screen-result-card" style={{marginBottom: '2rem'}}>
              <div className="stage-tag">🔍 Stage 1 Result</div>

              <div className={`screen-score-ring ${getRiskClass(screenResult.risk_label)}`}>
                <span className="screen-score-num">{screenResult.total_score}</span>
                <span className="screen-score-max">/ {screenResult.max_score}</span>
              </div>

              <div className={`screen-risk-label ${getRiskClass(screenResult.risk_label)}`}>
                {getRiskEmoji(screenResult.risk_label)} {screenResult.risk_label}
              </div>

              <p className="screen-triage-msg">{screenResult.triage_message}</p>

              {screenResult.domain_summary && (
                <div style={{textAlign: 'left', maxWidth: 480, margin: '0 auto 1.5rem'}}>
                  <p style={{fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.06em'}}>Domain Averages</p>
                  {Object.entries(screenResult.domain_summary).map(([d, v]) => (
                    <div key={d} className="domain-score-item" style={{marginBottom: '0.5rem'}}>
                      <div className="domain-score-header">
                        <span className="domain-score-name">{DOMAIN_ICONS[d] || '📊'} {d}</span>
                        <span className="domain-score-value">{v.avg} / 5</span>
                      </div>
                      <div className="score-bar-track">
                        <div className={`score-bar-fill ${v.avg >= 3.8 ? 'bar-high' : v.avg >= 2.6 ? 'bar-moderate' : 'bar-low'}`}
                             style={{width: `${(v.avg/5)*100}%`}} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="glass-panel" style={{marginBottom: '2rem'}}>
            <div className="section-header">
              <div className="stage-tag">📊 Stage 2</div>
              <h2 className="section-title">Detailed Assessment (25 Indicators)</h2>
              <p className="section-desc">
                Complete this comprehensive assessment to generate a full ML-based risk classification,
                SHAP-powered explanations, and evidence-grounded mentor action recommendations.
                Rate observations from your interactions over the past 2–4 weeks.
              </p>
            </div>

            <div className="questions-wrapper">
              {Object.entries(s2Groups).map(([domain, qs]) => (
                <div key={domain} className="domain-group">
                  <div className="domain-header">
                    <span className="domain-icon">{DOMAIN_ICONS[domain] || '📊'}</span>
                    <span className="domain-title">{domain}</span>
                    <span className="domain-count">{qs.length} question{qs.length > 1 ? 's' : ''}</span>
                  </div>
                  <div className="questions-list">
                    {qs.map((q) => (
                      <LikertQuestion
                        key={q.id}
                        question={q}
                        index={q.localIdx}
                        value={s2Answers[q.localIdx]}
                        onChange={handleS2Change}
                        globalIndex={q.localIdx}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="action-row">
            <button className="btn btn-secondary" onClick={() => setStep(1)}>← Redo Screening</button>
            <button id="btn-submit-stage2" className="btn btn-primary" onClick={submitStage2}>
              Generate Full Analysis →
            </button>
          </div>
        </div>
      )}

      {/* Loading for Stage 2 */}
      {loading && step === 2 && (
        <div className="glass-panel">
          <LoadingView steps={LOADING_STEPS} activeStep={loadingStep} />
        </div>
      )}

      {/* ── STEP 3: Full Results Dashboard ─────────────────── */}
      {step === 3 && fullResult && (
        <div style={{animation: 'fadeSlideUp 0.6s ease-out'}}>

          {/* Risk Hero Banner */}
          <div className={`risk-hero risk-${getRiskClass(fullResult.risk_label)}`}>
            <div className={`risk-badge-large`}>
              {getRiskEmoji(fullResult.risk_label)} Stage 2 Result
            </div>
            <div className="risk-level-text">{fullResult.risk_label}</div>
            {fullResult.student_name && fullResult.student_name !== 'Anonymous' && (
              <div className="risk-student-name">
                👤 {fullResult.student_name}
                {fullResult.student_id && ` · ${fullResult.student_id}`}
                {fullResult.programme && ` · ${fullResult.programme}`}
              </div>
            )}
          </div>

          <div className="results-grid">

            {/* ── XAI: Top Indicators ─────────────────────── */}
            <div className="result-card">
              <div className="result-card-header">
                <div className="result-card-icon">🔍</div>
                <div>
                  <div className="result-card-title">Key Distress Indicators</div>
                  <div className="result-card-subtitle">SHAP — Top contributing features</div>
                </div>
              </div>
              <ul className="indicators-list">
                {(fullResult.xai?.top_indicators || []).map((ind, i) => (
                  <li key={i} className="indicator-item">
                    <span className="indicator-dot" />
                    {ind}
                  </li>
                ))}
              </ul>
              {fullResult.xai?.consistency_check?.warnings?.length > 0 && (
                <div className="warning-box">
                  ⚠️ {fullResult.xai.consistency_check.warnings[0]}
                </div>
              )}
            </div>

            {/* ── Domain Breakdown ──────────────────────────── */}
            <div className="result-card">
              <div className="result-card-header">
                <div className="result-card-icon">📊</div>
                <div>
                  <div className="result-card-title">Domain Score Breakdown</div>
                  <div className="result-card-subtitle">Average observed severity per domain</div>
                </div>
              </div>
              <DomainScoreBars
                domainScores={fullResult.xai?.domain_scores || {}}
                riskClass={getRiskClass(fullResult.risk_label)}
              />
            </div>

            {/* ── Hybrid Classification ──────────────────────── */}
            <div className="result-card">
              <div className="result-card-header">
                <div className="result-card-icon">⚖️</div>
                <div>
                  <div className="result-card-title">Hybrid Classification</div>
                  <div className="result-card-subtitle">Rule-based · ML model · Conflict resolution</div>
                </div>
              </div>
              {fullResult.hybrid_decision && (
                <div className="hybrid-box">
                  <div className="hybrid-row">
                    <div className="hybrid-chip">📐 Rule-Based: {fullResult.hybrid_decision.rule_label}</div>
                    <div className="hybrid-chip">🤖 ML Model: {fullResult.hybrid_decision.ml_label}</div>
                    {fullResult.hybrid_decision.conflict
                      ? <span className="conflict-badge">⚡ Conflict</span>
                      : <span className="agree-badge">✅ Agreement</span>
                    }
                  </div>
                  <p style={{fontSize:'0.82rem', color:'var(--text-secondary)', lineHeight:1.6}}>
                    {fullResult.hybrid_decision.resolution}
                  </p>
                  <div className="score-summary">
                    <div className="hybrid-chip">Score: {fullResult.hybrid_decision.weighted_score} / {fullResult.hybrid_decision.max_weighted_score}</div>
                    <div className="hybrid-chip">Risk Ratio: {(fullResult.hybrid_decision.score_ratio * 100).toFixed(1)}%</div>
                  </div>
                </div>
              )}
            </div>

            {/* ── RAG Mentor Suggestion ─────────────────────── */}
            <div className="result-card">
              <div className="result-card-header">
                <div className="result-card-icon">💡</div>
                <div>
                  <div className="result-card-title">Suggested Mentor Actions</div>
                  <div className="result-card-subtitle">RAG — Evidence-grounded guidance</div>
                </div>
              </div>
              <p className="suggestion-text">
                {fullResult.rag?.knowledge_guidance || fullResult.rag?.suggested_action || '—'}
              </p>
              {fullResult.rag?.monitoring_recommendation && (
                <div className="monitoring-box">
                  <div className="monitoring-label">📋 Monitoring Recommendation</div>
                  {fullResult.rag.monitoring_recommendation}
                </div>
              )}
              {fullResult.rag?.verification?.issues?.length > 0 && (
                <div className="warning-box" style={{marginTop:'0.75rem'}}>
                  ⚠️ Verification note: {fullResult.rag.verification.issues[0]}
                </div>
              )}
            </div>

            {/* ── Research Context (RAG retrieved) ─────────── */}
            {fullResult.rag?.retrieved_context && (
              <div className="result-card full-width">
                <div className="result-card-header">
                  <div className="result-card-icon">📖</div>
                  <div>
                    <div className="result-card-title">Retrieved Research Context</div>
                    <div className="result-card-subtitle">RAG — Grounded from institutional research PDFs</div>
                  </div>
                </div>
                <p style={{
                  fontSize: '0.85rem',
                  color: 'var(--text-secondary)',
                  lineHeight: 1.75,
                  fontStyle: 'italic',
                  background: 'rgba(0,0,0,0.2)',
                  padding: '1rem',
                  borderRadius: 'var(--radius-md)',
                  borderLeft: '3px solid var(--accent)',
                }}>
                  "{fullResult.rag.retrieved_context}"
                </p>
              </div>
            )}

            {/* ── System Disclaimer ─────────────────────────── */}
            <FollowUpFeedbackCard
              currentUser={currentUser}
              fullResult={fullResult}
              studentName={studentName}
              studentId={studentId}
              studentEmail={studentEmail}
              programme={programme}
              generatedFeedbackLink={generatedFeedbackLink}
              onGenerateLink={generateStudentFeedbackLink}
              feedbackLinkSaving={feedbackLinkSaving}
              emailDeliveryStatus={emailDeliveryStatus}
            />

            <div className="result-card full-width" style={{
              background: 'rgba(0,0,0,0.15)',
              border: '1px solid rgba(255,255,255,0.05)',
            }}>
              <p style={{
                fontSize: '0.78rem',
                color: 'var(--text-muted)',
                textAlign: 'center',
                lineHeight: 1.7,
              }}>
                ⚠️ <strong style={{color:'var(--text-secondary)'}}>Important:</strong> This system is a decision-support tool for mentors only.
                It does not diagnose mental health conditions, provide therapy, or replace professional clinical judgment.
                All suggested actions are mentor-level interventions consistent with institutional advising protocols.
              </p>
            </div>
          </div>

          <div className="action-row">
            <button id="btn-new-assessment" className="btn btn-secondary" onClick={resetAll}>
              🔄 New Assessment
            </button>
            <button className="btn btn-primary" onClick={() => { setStep(2); scrollTop() }}>
              ← Revise Responses
            </button>
          </div>
        </div>
      )}

      {/* Loading for final step transition */}
      {loading && step > 2 && (
        <div className="glass-panel">
          <div className="loading-overlay">
            <div className="spinner" />
            <div className="loading-text">Processing…</div>
          </div>
        </div>
      )}
        </>
      </SignedIn>
    </div>
  )
}
