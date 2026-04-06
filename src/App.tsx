import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { SheetMusic } from './components/Canvas/SheetMusic';
import { useScoreStore } from './store/scoreStore';
import './App.css';
import { exportToPDF } from './utils/exportPDF';
import { BpmControl } from './components/Controls/BpmControl';
import { RecordButton } from './components/Controls/RecordButton';
import { PitchIndicator } from './components/Controls/PitchIndicator';
import { useMetronome } from './hooks/useMetronome';

/* ─── Icons ──────────────────────────────────────────────────────────────── */
const IconPencil = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
  </svg>
);

const IconBack = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="15 18 9 12 15 6" />
  </svg>
);

const IconSave = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
    <polyline points="17 21 17 13 7 13 7 21" />
    <polyline points="7 3 7 8 15 8" />
  </svg>
);

const IconExport = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7 10 12 15 17 10" />
    <line x1="12" y1="15" x2="12" y2="3" />
  </svg>
);

/* ─── App ────────────────────────────────────────────────────────────────── */
function App() {
  useMetronome();
  const navigate = useNavigate();

  const { clearScore, saveRecording, notes } = useScoreStore();

  const [title, setTitle] = useState(`Recording ${new Date().toLocaleString()}`);
  const [titleFocused, setTitleFocused] = useState(false);
  const [readyToSave, setReadyToSave] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    clearScore();
    setReadyToSave(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRecordingStopped = useCallback(() => {
    setReadyToSave(true);
  }, []);

  const handleSave = async () => {
    if (notes.length === 0) return;
    setSaving(true);
    const sheetId = await saveRecording(title.trim() || `Recording ${new Date().toLocaleString()}`);
    setSaving(false);
    if (sheetId) navigate('/dashboard');
  };

  const handleClear = () => {
    clearScore();
    setReadyToSave(false);
  };

  return (
    <div className="container">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="header">

        {/* Left — back link + editable title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: 0, flex: 1 }}>
          <Link
            to="/dashboard"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '3px',
              fontSize: '0.8rem',
              fontWeight: 600,
              color: '#B45309',
              textDecoration: 'none',
              flexShrink: 0,
              opacity: 0.8,
              transition: 'opacity 0.15s',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = '0.8')}
          >
            <IconBack />
            My Sheets
          </Link>

          {/* Vertical rule */}
          <span style={{ width: 1, height: 18, background: 'rgba(249,115,22,0.15)', flexShrink: 0 }} />

          {/* Editable title with pencil icon */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center', minWidth: 0, flex: 1, maxWidth: 360 }}>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onFocus={() => setTitleFocused(true)}
              onBlur={() => setTitleFocused(false)}
              placeholder="Recording name…"
              style={{
                fontSize: '0.95rem',
                fontWeight: 700,
                color: '#2C1A06',
                border: '1.5px solid',
                borderColor: titleFocused ? '#F97316' : 'transparent',
                borderRadius: '0.4rem',
                padding: '0.2rem 1.75rem 0.2rem 0.4rem',
                background: titleFocused ? 'rgba(249,115,22,0.04)' : 'transparent',
                outline: 'none',
                minWidth: 0,
                width: '100%',
                transition: 'border-color 0.18s, background 0.18s',
                fontFamily: 'inherit',
                boxShadow: titleFocused ? '0 0 0 3px rgba(249,115,22,0.1)' : 'none',
              }}
            />
            {/* Pencil icon — fades out when focused */}
            <span
              style={{
                position: 'absolute',
                right: '0.45rem',
                color: '#B45309',
                opacity: titleFocused ? 0 : 0.5,
                pointerEvents: 'none',
                transition: 'opacity 0.15s',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <IconPencil />
            </span>
          </div>
        </div>

        {/* Right — controls */}
        <div className="controls">

          {/* Group 1: recording tools */}
          <RecordButton mode="monitor" />
          <RecordButton onRecordingStopped={handleRecordingStopped} />
          <PitchIndicator />

          <span className="controls-divider" />

          {/* Group 2: tempo */}
          <BpmControl />

          <span className="controls-divider" />

          {/* Group 3: sheet actions */}
          <button className="export-btn" onClick={() => exportToPDF(title.trim() || 'sheet-music')}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <IconExport />
              Export PDF
            </span>
          </button>

          <button className="btn-danger" onClick={handleClear}>
            Clear
          </button>

          {readyToSave && (
            <button
              className="btn-primary"
              onClick={handleSave}
              disabled={saving || notes.length === 0}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <IconSave />
                {saving ? 'Saving…' : 'Save Sheet'}
              </span>
            </button>
          )}
        </div>
      </header>

      {/* ── Sheet music — fills remaining height ───────────────────────── */}
      <main className="main-content">
        <SheetMusic />
      </main>

    </div>
  );
}

export default App;
