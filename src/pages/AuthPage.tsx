import React, { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { supabase } from '../utils/supabase';
import { useAuth } from '../context/AuthContext';
import { BTN_ACCENT_BG, BTN_ACCENT_HOVER } from '../constants/theme';

const AuthPage: React.FC = () => {
  const { user } = useAuth();
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');

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
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#ffffff',
    }}>
      <div style={{
        background: 'white',
        border: '1px solid #e5e7eb',
        borderRadius: '0.75rem',
        padding: '2rem',
        width: '100%',
        maxWidth: '380px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      }}>
        <h1 style={{ margin: '0 0 0.25rem', fontSize: '1.5rem', color: '#111827' }}>
          Music Transcriber
        </h1>
        <p style={{ margin: '0 0 1.5rem', color: '#6b7280', fontSize: '0.875rem' }}>
          {isSignUp ? 'Create your account' : 'Sign in to your account'}
        </p>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            style={inputStyle}
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={inputStyle}
          />

          {error && (
            <p style={{ margin: 0, color: '#dc2626', fontSize: '0.875rem' }}>{error}</p>
          )}
          {successMsg && (
            <p style={{ margin: 0, color: '#16a34a', fontSize: '0.875rem' }}>{successMsg}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            style={{
              background: loading ? '#9ca3af' : BTN_ACCENT_BG,
              color: 'white',
              border: 'none',
              borderRadius: '0.375rem',
              padding: '0.6rem 1rem',
              fontSize: '0.875rem',
              fontWeight: 500,
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
            onMouseEnter={(e) => { if (!loading) (e.target as HTMLButtonElement).style.background = BTN_ACCENT_HOVER; }}
            onMouseLeave={(e) => { if (!loading) (e.target as HTMLButtonElement).style.background = BTN_ACCENT_BG; }}
          >
            {loading ? 'Loading...' : isSignUp ? 'Create Account' : 'Sign In'}
          </button>
        </form>

        <p style={{ margin: '1.25rem 0 0', textAlign: 'center', fontSize: '0.875rem', color: '#6b7280' }}>
          {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
          <button
            onClick={() => { setIsSignUp(!isSignUp); setError(''); setSuccessMsg(''); }}
            style={{
              background: 'none',
              border: 'none',
              padding: 0,
              color: BTN_ACCENT_BG,
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontWeight: 500,
            }}
          >
            {isSignUp ? 'Sign in' : 'Sign up'}
          </button>
        </p>
      </div>
    </div>
  );
};

const inputStyle: React.CSSProperties = {
  border: '1px solid #d1d5db',
  borderRadius: '0.375rem',
  padding: '0.5rem 0.75rem',
  fontSize: '0.875rem',
  outline: 'none',
  color: '#111827',
  background: 'white',
};

export default AuthPage;
