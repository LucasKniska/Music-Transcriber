import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { SheetMusic } from './components/Canvas/SheetMusic';
import { useScoreStore } from './store/scoreStore';
import './App.css';
import { exportToPDF } from './utils/exportPDF';
import { BpmControl } from './components/Controls/BpmControl';
import { RecordButton } from './components/Controls/RecordButton';
import { useMetronome } from './hooks/useMetronome';
import { BTN_ACCENT_BG, BTN_ACCENT_HOVER } from './constants/theme';

function App() {
  useMetronome();
  const navigate = useNavigate();

  const { clearScore, saveRecording, notes } = useScoreStore();

  const [title, setTitle] = useState(`Recording ${new Date().toLocaleString()}`);
  const [readyToSave, setReadyToSave] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveBtnHovered, setSaveBtnHovered] = useState(false);

  useEffect(() => {
    clearScore();
    setReadyToSave(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRecordingStopped = () => {
    setReadyToSave(true);
  };

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
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flex: 1, minWidth: 0 }}>
          <Link to="/dashboard" style={{ fontSize: '0.875rem', color: '#9ca3af', flexShrink: 0 }}>
            ← My Sheets
          </Link>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Recording name..."
            style={{
              fontSize: '1.1rem',
              fontWeight: 600,
              color: '#111827',
              border: '1px solid transparent',
              borderRadius: '0.25rem',
              padding: '0.125rem 0.375rem',
              background: 'transparent',
              outline: 'none',
              minWidth: 0,
              flex: 1,
              maxWidth: '360px',
              transition: 'border-color 0.15s',
            }}
            onFocus={(e) => (e.target.style.borderColor = '#fed7aa')}
            onBlur={(e) => (e.target.style.borderColor = 'transparent')}
          />
        </div>

        <div className="controls">
          <RecordButton onRecordingStopped={handleRecordingStopped} />
          <BpmControl />
          <button className="btn-danger" onClick={handleClear}>
            Clear Sheet
          </button>
          <button className="export-btn" onClick={() => exportToPDF()}>
            Export PDF
          </button>
          {readyToSave && (
            <button
              onClick={handleSave}
              disabled={saving || notes.length === 0}
              onMouseEnter={() => setSaveBtnHovered(true)}
              onMouseLeave={() => setSaveBtnHovered(false)}
              style={{
                background: saving ? '#9ca3af' : saveBtnHovered ? BTN_ACCENT_HOVER : BTN_ACCENT_BG,
                color: 'white',
                border: 'none',
                fontWeight: 600,
              }}
            >
              {saving ? 'Saving...' : 'Save Sheet'}
            </button>
          )}
        </div>
      </header>

      <main className="main-content">
        <SheetMusic />
      </main>
    </div>
  );
}

export default App;
