import { useEffect } from 'react';
import { SheetMusic } from './components/Canvas/SheetMusic';
import { useScoreStore } from './store/scoreStore';
import './App.css';
import { BpmControl } from './components/Controls/BpmControl';
import { RecordButton } from './components/Controls/RecordButton';
import { PitchIndicator } from './components/Controls/PitchIndicator';
import { useMetronome } from './hooks/useMetronome';

function App() {
  useMetronome();

  const { clearScore, insertionPointNoteId, clearInsertionPoint } = useScoreStore();

  useEffect(() => {
    clearScore();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="container">

      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: 1 }}>
          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#2C1A06' }}>
            Music Transcriber
          </span>
        </div>

        <div className="controls">
          <RecordButton mode="monitor" />
          <RecordButton />
          <PitchIndicator />

          <span className="controls-divider" />

          <BpmControl />

          <span className="controls-divider" />

          <button className="btn-danger" onClick={() => clearScore()}>
            Clear
          </button>

          {insertionPointNoteId && (
            <>
              <span className="controls-divider" />
              <span style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                padding: '0.25rem 0.65rem',
                background: 'rgba(249,115,22,0.1)',
                border: '1px solid rgba(249,115,22,0.25)',
                borderRadius: '999px',
                fontSize: '0.72rem',
                fontWeight: 600,
                color: '#92400E',
                whiteSpace: 'nowrap',
              }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#F97316', flexShrink: 0 }} />
                Marker set
                <button
                  onClick={() => clearInsertionPoint()}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 16,
                    height: 16,
                    padding: 0,
                    border: 'none',
                    borderRadius: '50%',
                    background: 'rgba(249,115,22,0.15)',
                    color: '#92400E',
                    fontSize: '0.65rem',
                    cursor: 'pointer',
                    lineHeight: 1,
                  }}
                  title="Clear insertion point"
                >
                  ✕
                </button>
              </span>
            </>
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
