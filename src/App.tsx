import { SheetMusic } from './components/Canvas/SheetMusic';
import { useTranscriber } from './hooks/useTranscriber';
import { useScoreStore } from './store/scoreStore';
import './App.css';
import { exportToPDF } from './utils/exportPDF';
import { BpmControl } from './components/Controls/BpmControl';
import { RecordButton } from './components/Controls/RecordButton'; // Import the new button
import { useMetronome } from './hooks/useMetronome';

function App() {
  // Activate the Logic Engine (Keyboard listeners)
  useTranscriber();
  useMetronome();

  // Get the clear function
  const clearScore = useScoreStore((state) => state.clearScore);

  return (
    <div className="container">
      <header className="header">
        <h1>Drum Transcriber</h1>
        <div className="controls">
          {/* New Record Button placed here */}
          <RecordButton />
          
          <BpmControl />
          
          <button 
            onClick={clearScore}
            className="px-3 py-1 text-sm text-red-600 border border-red-200 rounded hover:bg-red-50"
          >
            Clear Sheet
          </button>
          <button
            onClick={() => exportToPDF()}
            className="export-btn"
          >
            Export PDF
          </button>
        </div>
      </header>

      <main className="main-content">
        {/* Removed the old AudioTranscriber div */}
        <SheetMusic />
      </main>
    </div>
  );
}

export default App;