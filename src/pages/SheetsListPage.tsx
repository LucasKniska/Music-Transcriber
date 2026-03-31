import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { supabase } from '../utils/supabase';
import { useAuth } from '../context/AuthContext';
import type { Sheet } from '../types';
import { BTN_ACCENT_BG, BTN_ACCENT_HOVER } from '../constants/theme';

/* ─── Inline SVGs ─────────────────────────────────────────────────────────── */

const LogoWave = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#F97316" strokeWidth="2.5" strokeLinecap="round">
    <path d="M2 12 Q 5 6, 8 12 Q 11 18, 14 12 Q 17 6, 20 12 Q 22 15, 24 12" />
  </svg>
);

const IconSheets = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" />
  </svg>
);

const IconSparkle = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5z" />
    <path d="M19 3l.75 2.25L22 6l-2.25.75L19 9l-.75-2.25L16 6l2.25-.75z" />
    <path d="M5 17l.5 1.5L7 19l-1.5.5L5 21l-.5-1.5L3 19l1.5-.5z" />
  </svg>
);

const IconLock = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
);

const IconMusicNote = () => (
  <svg width="72" height="72" viewBox="0 0 24 24" fill="none" stroke="#F97316" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.12">
    <path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" />
  </svg>
);

const IconEmptyLibrary = () => (
  <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#F97316" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" />
    <line x1="1" y1="1" x2="23" y2="23" stroke="#F97316" strokeWidth="1.4" />
  </svg>
);

const IconTrash = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    <path d="M10 11v6" /><path d="M14 11v6" /><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
  </svg>
);

/* ─── Skeleton Card ───────────────────────────────────────────────────────── */

const SkeletonCard = () => (
  <div className="db-card">
    <div className="db-card-strip" style={{ background: 'linear-gradient(135deg, #F5ECD6, #EDE0C4)' }} />
    <div style={{ padding: '1rem 1.25rem' }}>
      <div className="db-skeleton db-skeleton-title" />
      <div className="db-skeleton db-skeleton-meta" />
    </div>
  </div>
);

/* ─── Main Component ──────────────────────────────────────────────────────── */

const SheetsListPage: React.FC = () => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [sheets, setSheets] = useState<Sheet[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [copilotHint, setCopilotHint] = useState(false);
  const [hintTimer, setHintTimer] = useState<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!user) return;
    supabase
      .from('sheets')
      .select('id, user_id, title, bpm, notes, created_at, updated_at')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false })
      .then(({ data, error }) => {
        if (!error && data) setSheets(data as Sheet[]);
        setLoading(false);
      });
  }, [user]);

  const handleDelete = async (e: React.MouseEvent, id: string, title: string) => {
    e.preventDefault();
    if (!confirm(`Delete "${title}"? This cannot be undone.`)) return;
    setDeletingId(id);
    const { error } = await supabase.from('sheets').delete().eq('id', id);
    if (!error) setSheets((prev) => prev.filter((s) => s.id !== id));
    setDeletingId(null);
  };

  const handleCopilotClick = () => {
    if (hintTimer) clearTimeout(hintTimer);
    setCopilotHint(true);
    const t = setTimeout(() => setCopilotHint(false), 3200);
    setHintTimer(t);
  };

  const userInitial = user?.email?.[0]?.toUpperCase() ?? '?';
  const userEmail = user?.email ?? '';

  return (
    <>
      <style>{`
        .db-root {
          min-height: 100vh;
          background: #FDF6EC;
          font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
          display: flex;
          flex-direction: column;
          box-sizing: border-box;
        }

        /* ── Shell ── */
        .db-shell {
          flex: 1;
          display: flex;
          overflow: hidden;
          min-height: 100vh;
        }

        /* ── Sidebar ── */
        .db-sidebar {
          width: 220px;
          flex-shrink: 0;
          background: #F0E8D8;
          display: flex;
          flex-direction: column;
          padding: 0;
          border-right: 1px solid rgba(249,115,22,0.1);
        }
        .db-sidebar-logo {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 1.4rem 1.25rem 1.2rem;
          text-decoration: none;
          border-bottom: 1px solid rgba(249,115,22,0.1);
          margin-bottom: 0.5rem;
        }
        .db-sidebar-logo-text {
          font-size: 1rem;
          font-weight: 800;
          color: #2C1A06;
          letter-spacing: -0.3px;
        }
        .db-beta {
          font-size: 0.55rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          background: rgba(251,191,36,0.2);
          color: #92400E;
          border: 1px solid rgba(180,83,9,0.18);
          border-radius: 999px;
          padding: 0.15rem 0.45rem;
        }

        .db-nav-section {
          padding: 0 0.75rem;
          margin-bottom: 0.25rem;
        }
        .db-nav-item {
          display: flex;
          align-items: center;
          gap: 0.625rem;
          padding: 0.55rem 0.75rem;
          border-radius: 0.625rem;
          font-size: 0.875rem;
          font-weight: 600;
          color: #78350F;
          text-decoration: none;
          cursor: pointer;
          transition: background 0.15s;
          user-select: none;
        }
        .db-nav-item:hover {
          background: rgba(249,115,22,0.08);
        }
        .db-nav-item.active {
          background: rgba(249,115,22,0.12);
          color: #C2410C;
          border-left: 3px solid #F97316;
          padding-left: calc(0.75rem - 3px);
        }
        .db-nav-item.locked {
          opacity: 0.65;
          cursor: default;
        }
        .db-nav-item.locked:hover {
          background: rgba(249,115,22,0.05);
        }
        .db-nav-item-spacer { flex: 1; }
        .db-nav-lock {
          margin-left: auto;
          color: #B45309;
          opacity: 0.6;
        }

        .db-coming-soon-label {
          font-size: 0.62rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: #B45309;
          padding: 0.9rem 1.5rem 0.35rem;
          opacity: 0.7;
        }

        .db-copilot-hint {
          margin: 0 0.75rem;
          padding: 0.6rem 0.75rem;
          background: rgba(249,115,22,0.07);
          border: 1px solid rgba(249,115,22,0.15);
          border-radius: 0.625rem;
          font-size: 0.78rem;
          color: #92400E;
          line-height: 1.5;
          font-weight: 500;
          animation: hint-in 0.22s ease;
          transition: opacity 0.3s;
        }
        .db-copilot-hint.hiding {
          opacity: 0;
        }
        @keyframes hint-in {
          from { opacity: 0; transform: translateY(-4px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        .db-sidebar-spacer { flex: 1; }

        .db-sidebar-bottom {
          padding: 1rem 1.25rem;
          border-top: 1px solid rgba(249,115,22,0.1);
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        .db-user-row {
          display: flex;
          align-items: center;
          gap: 0.625rem;
        }
        .db-avatar {
          width: 28px;
          height: 28px;
          border-radius: 50%;
          background: linear-gradient(135deg, #F97316, #C2410C);
          color: white;
          font-size: 0.7rem;
          font-weight: 700;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }
        .db-user-email {
          font-size: 0.78rem;
          color: #78350F;
          font-weight: 500;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          min-width: 0;
        }
        .db-signout-btn {
          font-size: 0.78rem;
          font-weight: 600;
          color: #B45309;
          background: none;
          border: none;
          padding: 0;
          cursor: pointer;
          text-align: left;
          font-family: inherit;
          transition: color 0.15s;
          opacity: 0.7;
        }
        .db-signout-btn:hover { opacity: 1; color: #C2410C; }

        /* ── Main content ── */
        .db-main {
          flex: 1;
          background: #FDF6EC;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
        }
        .db-main-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 2rem 2.5rem 1.5rem;
          border-bottom: 1px solid rgba(249,115,22,0.08);
          position: sticky;
          top: 0;
          background: rgba(253,246,236,0.92);
          backdrop-filter: blur(12px);
          z-index: 10;
        }
        .db-main-title {
          font-size: 1.5rem;
          font-weight: 800;
          color: #2C1A06;
          letter-spacing: -0.03em;
        }
        .db-main-subtitle {
          font-size: 0.825rem;
          color: #B45309;
          font-weight: 400;
          margin-top: 0.2rem;
        }
        .db-new-btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: #F97316;
          color: white;
          border: none;
          border-radius: 999px;
          padding: 0.55rem 1.25rem;
          font-size: 0.875rem;
          font-weight: 700;
          font-family: inherit;
          cursor: pointer;
          transition: background 0.15s, transform 0.15s, box-shadow 0.15s;
          box-shadow: 0 2px 12px rgba(249,115,22,0.3);
          letter-spacing: 0.01em;
        }
        .db-new-btn:hover {
          background: #C2410C;
          transform: translateY(-1px);
          box-shadow: 0 6px 20px rgba(249,115,22,0.38);
        }

        .db-content {
          padding: 2rem 2.5rem;
          flex: 1;
        }

        /* ── Cards grid ── */
        .db-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
          gap: 1.25rem;
        }

        .db-card {
          background: #FEFAF3;
          border: 1px solid rgba(249,115,22,0.11);
          border-radius: 1.1rem;
          overflow: hidden;
          text-decoration: none;
          display: block;
          transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s;
          position: relative;
        }
        .db-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 10px 32px rgba(249,115,22,0.13);
          border-color: rgba(249,115,22,0.25);
        }
        .db-card-strip {
          height: 68px;
          background: linear-gradient(135deg, #FFF0D6 0%, #FFE0B2 100%);
          display: flex;
          align-items: center;
          justify-content: flex-end;
          padding-right: 1rem;
          overflow: hidden;
          position: relative;
        }
        .db-card-strip-watermark {
          position: absolute;
          right: -4px;
          bottom: -12px;
          opacity: 0.18;
          pointer-events: none;
        }
        .db-card-body {
          padding: 0.875rem 1.1rem 0.75rem;
        }
        .db-card-title {
          font-size: 0.9rem;
          font-weight: 700;
          color: #2C1A06;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          margin-bottom: 0.35rem;
        }
        .db-card-meta {
          font-size: 0.75rem;
          color: #B45309;
          font-weight: 400;
          margin-bottom: 0.75rem;
        }
        .db-card-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-top: 0.5rem;
          border-top: 1px solid rgba(249,115,22,0.08);
        }
        .db-card-open {
          font-size: 0.78rem;
          font-weight: 600;
          color: #F97316;
          letter-spacing: 0.01em;
        }
        .db-card-delete {
          display: flex;
          align-items: center;
          gap: 4px;
          background: none;
          border: none;
          font-size: 0.75rem;
          font-weight: 500;
          color: #B45309;
          font-family: inherit;
          cursor: pointer;
          padding: 0.2rem 0.4rem;
          border-radius: 0.3rem;
          opacity: 0;
          transition: opacity 0.15s, background 0.15s, color 0.15s;
        }
        .db-card:hover .db-card-delete {
          opacity: 1;
        }
        .db-card-delete:hover {
          background: rgba(220,38,38,0.07);
          color: #dc2626;
        }

        /* ── Skeleton ── */
        .db-skeleton {
          background: linear-gradient(90deg, #EDE0C4 25%, #F5ECD6 50%, #EDE0C4 75%);
          background-size: 200% 100%;
          animation: shimmer 1.5s infinite;
          border-radius: 0.375rem;
        }
        @keyframes shimmer {
          0%   { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
        .db-skeleton-title { height: 14px; width: 70%; margin-bottom: 8px; }
        .db-skeleton-meta  { height: 11px; width: 50%; }

        /* ── Empty state ── */
        .db-empty {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 4rem 2rem;
          gap: 0;
        }
        .db-empty-icon {
          margin-bottom: 1.25rem;
          opacity: 0.6;
        }
        .db-empty-title {
          font-size: 1.25rem;
          font-weight: 700;
          color: #2C1A06;
          margin-bottom: 0.5rem;
          letter-spacing: -0.02em;
        }
        .db-empty-sub {
          font-size: 0.9rem;
          color: #92400E;
          font-weight: 400;
          max-width: 340px;
          line-height: 1.6;
          margin-bottom: 2rem;
        }
      `}</style>

      <div className="db-root">
        <div className="db-shell">

          {/* ── Sidebar ──────────────────────────────────────────────────── */}
          <aside className="db-sidebar">

            {/* Logo */}
            <Link to="/" className="db-sidebar-logo">
              <LogoWave />
              <span className="db-sidebar-logo-text">Score AI</span>
              <span className="db-beta">Beta</span>
            </Link>

            {/* Primary nav */}
            <div className="db-nav-section">
              <div className="db-nav-item active">
                <IconSheets />
                My Sheets
              </div>
            </div>

            {/* Score Copilot — locked */}
            <p className="db-coming-soon-label">Coming Soon</p>
            <div className="db-nav-section">
              <div
                className="db-nav-item locked"
                onClick={handleCopilotClick}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && handleCopilotClick()}
              >
                <IconSparkle />
                Score Copilot
                <span className="db-nav-lock"><IconLock /></span>
              </div>
            </div>

            {/* Inline hint */}
            {copilotHint && (
              <div className={`db-copilot-hint${!copilotHint ? ' hiding' : ''}`}>
                We're building something powerful. Stay tuned.
              </div>
            )}

            <div className="db-sidebar-spacer" />

            {/* User + sign out */}
            <div className="db-sidebar-bottom">
              <div className="db-user-row">
                <div className="db-avatar">{userInitial}</div>
                <span className="db-user-email" title={userEmail}>{userEmail}</span>
              </div>
              <button className="db-signout-btn" onClick={signOut}>
                Sign out
              </button>
            </div>
          </aside>

          {/* ── Main content ─────────────────────────────────────────────── */}
          <main className="db-main">

            {/* Sticky header */}
            <div className="db-main-header">
              <div>
                <div className="db-main-title">Your Library</div>
                {!loading && (
                  <div className="db-main-subtitle">
                    {sheets.length === 0
                      ? 'No recordings yet'
                      : `${sheets.length} recording${sheets.length !== 1 ? 's' : ''}`}
                  </div>
                )}
              </div>
              <button className="db-new-btn" onClick={() => navigate('/record')}>
                + New Recording
              </button>
            </div>

            {/* Content area */}
            <div className="db-content">
              {loading ? (
                <div className="db-grid">
                  <SkeletonCard />
                  <SkeletonCard />
                  <SkeletonCard />
                </div>
              ) : sheets.length === 0 ? (
                <div className="db-empty">
                  <div className="db-empty-icon">
                    <IconEmptyLibrary />
                  </div>
                  <div className="db-empty-title">Your library is empty</div>
                  <p className="db-empty-sub">
                    Record your first performance and watch it become sheet music in real time.
                  </p>
                  <button className="db-new-btn" onClick={() => navigate('/record')}>
                    + Start Recording
                  </button>
                </div>
              ) : (
                <div className="db-grid">
                  {sheets.map((sheet) => (
                    <Link
                      key={sheet.id}
                      to={`/sheet/${sheet.id}`}
                      className="db-card"
                    >
                      <div className="db-card-strip">
                        <div className="db-card-strip-watermark">
                          <IconMusicNote />
                        </div>
                      </div>
                      <div className="db-card-body">
                        <div className="db-card-title" title={sheet.title}>
                          {sheet.title}
                        </div>
                        <div className="db-card-meta">
                          {sheet.notes.length} note{sheet.notes.length !== 1 ? 's' : ''}
                          &nbsp;·&nbsp;{sheet.bpm} BPM
                          &nbsp;·&nbsp;{new Date(sheet.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                        </div>
                        <div className="db-card-footer">
                          <span className="db-card-open">Open →</span>
                          <button
                            className="db-card-delete"
                            onClick={(e) => handleDelete(e, sheet.id, sheet.title)}
                            disabled={deletingId === sheet.id}
                            title="Delete sheet"
                          >
                            <IconTrash />
                            {deletingId === sheet.id ? 'Deleting…' : 'Delete'}
                          </button>
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </main>

        </div>
      </div>
    </>
  );
};

export default SheetsListPage;
