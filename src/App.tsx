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

  const { clearScore } = useScoreStore();

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
        </div>
      </header>

      <main className="main-content">
        <SheetMusic />
      </main>

    </div>
  );
}

export default App;
