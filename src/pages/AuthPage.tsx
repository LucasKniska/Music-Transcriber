import React, { useState } from 'react';
import { Navigate, Link } from 'react-router-dom';
import { supabase } from '../utils/supabase';
import { useAuth } from '../context/AuthContext';

/* ─── Waveform decoration ─────────────────────────────────────────────────── */
const MiniWave: React.FC = () => (
  <svg viewBox="0 0 400 60" preserveAspectRatio="none" style={{ width: '100%', height: 60, display: 'block', opacity: 0.22 }}>
    <path d="M0 30 C33 10, 67 50, 100 30 C133 10, 167 50, 200 30 C233 10, 267 50, 300 30 C333 10, 367 50, 400 30"
      stroke="#F97316" strokeWidth="2.5" fill="none" />
    <path d="M0 30 C25 18, 50 42, 75 30 C100 18, 125 42, 150 30 C175 18, 200 42, 225 30 C250 18, 275 42, 300 30 C325 18, 350 42, 375 30 C387 24, 393 36, 400 30"
      stroke="#F97316" strokeWidth="1.5" fill="none" opacity="0.6" />
  </svg>
);

const LogoWave = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#F97316" strokeWidth="2.5" strokeLinecap="round">
    <path d="M2 12 Q 5 6, 8 12 Q 11 18, 14 12 Q 17 6, 20 12 Q 22 15, 24 12" />
  </svg>
);

const Spinner = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style={{ animation: 'spin 0.8s linear infinite' }}>
    <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" strokeWidth="3" />
    <path d="M12 2 A10 10 0 0 1 22 12" stroke="white" strokeWidth="3" strokeLinecap="round" />
  </svg>
);

/* ─── Auth Page ───────────────────────────────────────────────────────────── */

const AuthPage: React.FC = () => {
  const { user } = useAuth();
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');
  const [, setFocusedField] = useState<string | null>(null);

  if (user) return <Navigate to="/dashboard" replace />;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccessMsg('');
    setLoading(true);

    if (isSignUp) {
      const { error } = await supabase.auth.signUp({ email, password });
      if (error) {
        setError(error.message);
      } else {
        setSuccessMsg('Account created! Check your email to confirm, then sign in.');
      }
    } else {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) setError(error.message);
    }

    setLoading(false);
  };

  return (
    <>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes auth-fade-up {
          from { opacity: 0; transform: translateY(18px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes float-note {
          0%, 100% { transform: translateY(0) rotate(-8deg); opacity: 0.06; }
          50%       { transform: translateY(-14px) rotate(4deg); opacity: 0.1; }
        }

        .auth-root {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: radial-gradient(ellipse 90% 70% at 50% 30%, #FFF0D6 0%, #FDF6EC 65%);
          font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
          padding: 2rem 1.25rem;
          position: relative;
          overflow: hidden;
        }

        .auth-bg-note {
          position: absolute;
          pointer-events: none;
          animation: float-note 7s ease-in-out infinite;
          color: #F97316;
          font-size: 5rem;
          user-select: none;
        }

        .auth-card {
          background: #FEFAF3;
          border: 1px solid rgba(249,115,22,0.14);
          border-radius: 1.5rem;
          width: 100%;
          max-width: 400px;
          box-shadow: 0 8px 48px rgba(44,26,6,0.09), 0 2px 12px rgba(249,115,22,0.08);
          overflow: hidden;
          animation: auth-fade-up 0.45s cubic-bezier(0.22,1,0.36,1) both;
          position: relative;
          z-index: 1;
        }

        .auth-card-header {
          background: linear-gradient(135deg, #FFF0D6 0%, #FFE8C0 100%);
          border-bottom: 1px solid rgba(249,115,22,0.12);
          padding: 1.75rem 2rem 1.5rem;
          position: relative;
          overflow: hidden;
        }

        .auth-card-header-wave {
          position: absolute;
          bottom: 0; left: 0; right: 0;
        }

        .auth-logo-row {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 1rem;
          text-decoration: none;
        }

        .auth-logo-text {
          font-size: 1.15rem;
          font-weight: 800;
          color: #2C1A06;
          letter-spacing: -0.3px;
        }

        .auth-beta {
          font-size: 0.58rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          background: rgba(251,191,36,0.22);
          color: #92400E;
          border: 1px solid rgba(180,83,9,0.18);
          border-radius: 999px;
          padding: 0.15rem 0.45rem;
        }

        .auth-heading {
          font-size: 1.45rem;
          font-weight: 800;
          color: #2C1A06;
          letter-spacing: -0.025em;
          margin: 0 0 0.3rem;
          line-height: 1.2;
        }

        .auth-sub {
          font-size: 0.875rem;
          color: #92400E;
          font-weight: 400;
          margin: 0;
          line-height: 1.5;
        }

        .auth-form-area {
          padding: 1.75rem 2rem 2rem;
        }

        .auth-field {
          display: flex;
          flex-direction: column;
          gap: 0.35rem;
          margin-bottom: 1rem;
        }

        .auth-label {
          font-size: 0.8rem;
          font-weight: 600;
          color: #78350F;
          letter-spacing: 0.01em;
        }

        .auth-input {
          border: 1.5px solid rgba(249,115,22,0.2);
          border-radius: 0.6rem;
          padding: 0.6rem 0.875rem;
          font-size: 0.9rem;
          outline: none;
          color: #2C1A06;
          background: #FFFBF5;
          font-family: inherit;
          transition: border-color 0.18s, box-shadow 0.18s;
        }

        .auth-input:focus {
          border-color: #F97316;
          box-shadow: 0 0 0 3px rgba(249,115,22,0.12);
        }

        .auth-input::placeholder {
          color: #C4956A;
          opacity: 0.7;
        }

        .auth-error {
          display: flex;
          align-items: flex-start;
          gap: 6px;
          background: rgba(220,38,38,0.06);
          border: 1px solid rgba(220,38,38,0.15);
          border-radius: 0.5rem;
          padding: 0.6rem 0.75rem;
          font-size: 0.825rem;
          color: #dc2626;
          line-height: 1.4;
          margin-bottom: 1rem;
        }

        .auth-success {
          display: flex;
          align-items: flex-start;
          gap: 6px;
          background: rgba(22,163,74,0.06);
          border: 1px solid rgba(22,163,74,0.2);
          border-radius: 0.5rem;
          padding: 0.6rem 0.75rem;
          font-size: 0.825rem;
          color: #15803d;
          line-height: 1.4;
          margin-bottom: 1rem;
        }

        .auth-submit-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          background: #F97316;
          color: white;
          border: none;
          border-radius: 0.625rem;
          padding: 0.7rem 1.25rem;
          font-size: 0.9rem;
          font-weight: 700;
          font-family: inherit;
          cursor: pointer;
          transition: background 0.18s, transform 0.18s, box-shadow 0.18s;
          box-shadow: 0 3px 16px rgba(249,115,22,0.32);
          letter-spacing: 0.01em;
          margin-top: 0.25rem;
        }

        .auth-submit-btn:hover:not(:disabled) {
          background: #C2410C;
          transform: translateY(-1px);
          box-shadow: 0 6px 22px rgba(249,115,22,0.4);
        }

        .auth-submit-btn:active:not(:disabled) {
          transform: translateY(0);
        }

        .auth-submit-btn:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }

        .auth-toggle-row {
          margin-top: 1.5rem;
          text-align: center;
          font-size: 0.85rem;
          color: #92400E;
        }

        .auth-toggle-btn {
          background: none;
          border: none;
          padding: 0;
          color: #F97316;
          font-weight: 700;
          font-size: 0.85rem;
          font-family: inherit;
          cursor: pointer;
          text-decoration: underline;
          text-decoration-color: rgba(249,115,22,0.3);
          transition: color 0.15s;
        }

        .auth-toggle-btn:hover {
          color: #C2410C;
        }

        .auth-trust-row {
          margin-top: 1.25rem;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          opacity: 0.55;
        }

        .auth-trust-item {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 0.72rem;
          color: #92400E;
          font-weight: 500;
        }

        .auth-divider {
          width: 1px;
          height: 12px;
          background: rgba(180,83,9,0.25);
        }
      `}</style>

      <div className="auth-root">
        {/* Floating background notes */}
        <span className="auth-bg-note" style={{ left: '8%', top: '15%', animationDelay: '0s' }}>♪</span>
        <span className="auth-bg-note" style={{ right: '10%', top: '20%', animationDelay: '2.5s', fontSize: '3.5rem' }}>♩</span>
        <span className="auth-bg-note" style={{ left: '15%', bottom: '20%', animationDelay: '1.2s', fontSize: '4rem' }}>♫</span>
        <span className="auth-bg-note" style={{ right: '8%', bottom: '25%', animationDelay: '3.8s', fontSize: '3rem' }}>♬</span>

        <div className="auth-card">
          {/* Header */}
          <div className="auth-card-header">
            <Link to="/" className="auth-logo-row">
              <LogoWave />
              <span className="auth-logo-text">Score AI</span>
              <span className="auth-beta">Beta</span>
            </Link>
            <h1 className="auth-heading">
              {isSignUp ? 'Start your journey' : 'Welcome back'}
            </h1>
            <p className="auth-sub">
              {isSignUp
                ? 'Create your free account and start transcribing music in seconds.'
                : 'Your music is waiting. Sign in to continue.'}
            </p>
            <div className="auth-card-header-wave">
              <MiniWave />
            </div>
          </div>

          {/* Form */}
          <div className="auth-form-area">
            <form onSubmit={handleSubmit}>
              <div className="auth-field">
                <label className="auth-label" htmlFor="auth-email">Email address</label>
                <input
                  id="auth-email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onFocus={() => setFocusedField('email')}
                  onBlur={() => setFocusedField(null)}
                  required
                  className="auth-input"
                />
              </div>

              <div className="auth-field">
                <label className="auth-label" htmlFor="auth-password">Password</label>
                <input
                  id="auth-password"
                  type="password"
                  placeholder={isSignUp ? 'At least 6 characters' : '••••••••'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onFocus={() => setFocusedField('password')}
                  onBlur={() => setFocusedField(null)}
                  required
                  className="auth-input"
                />
              </div>

              {error && (
                <div className="auth-error">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style={{ flexShrink: 0, marginTop: 1 }}>
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                  </svg>
                  {error}
                </div>
              )}
              {successMsg && (
                <div className="auth-success">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style={{ flexShrink: 0, marginTop: 1 }}>
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                  </svg>
                  {successMsg}
                </div>
              )}

              <button type="submit" disabled={loading} className="auth-submit-btn">
                {loading && <Spinner />}
                {loading ? 'Just a moment…' : isSignUp ? 'Create Free Account →' : 'Sign In →'}
              </button>
            </form>

            <div className="auth-toggle-row">
              {isSignUp ? 'Already have an account? ' : "Don't have an account? "}
              <button
                onClick={() => { setIsSignUp(!isSignUp); setError(''); setSuccessMsg(''); }}
                className="auth-toggle-btn"
              >
                {isSignUp ? 'Sign in' : 'Sign up free'}
              </button>
            </div>

            <div className="auth-trust-row">
              <span className="auth-trust-item">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="#92400E"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z"/></svg>
                Secure
              </span>
              <span className="auth-divider" />
              <span className="auth-trust-item">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="#92400E"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
                Free forever
              </span>
              <span className="auth-divider" />
              <span className="auth-trust-item">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="#92400E"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/></svg>
                No card needed
              </span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default AuthPage;
