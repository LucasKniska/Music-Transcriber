import React from 'react';
import { useScoreStore } from '../../store/scoreStore';

export const BpmControl: React.FC = () => {
  const bpm            = useScoreStore((s) => s.bpm);
  const setBpm         = useScoreStore((s) => s.setBpm);
  const isMetronomeOn  = useScoreStore((s) => s.isMetronomeOn);
  const toggleMetronome = useScoreStore((s) => s.toggleMetronome);

  return (
    <>
      <style>{`
        .bpm-root {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        .bpm-toggle {
          width: 26px;
          height: 26px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 0.65rem;
          padding: 0;
          flex-shrink: 0;
          transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
        }
        .bpm-toggle.on {
          background: #dc2626 !important;
          border-color: #dc2626 !important;
          color: white !important;
          box-shadow: 0 0 0 3px rgba(220,38,38,0.15);
        }
        .bpm-toggle.on:hover {
          background: #b91c1c !important;
        }
        .bpm-label {
          font-size: 0.78rem;
          font-weight: 700;
          color: #92400E;
          white-space: nowrap;
          min-width: 52px;
          font-family: inherit;
          font-variant-numeric: tabular-nums;
        }
        .bpm-slider {
          -webkit-appearance: none;
          appearance: none;
          width: 80px;
          height: 3px;
          border-radius: 999px;
          background: linear-gradient(
            to right,
            #F97316 0%,
            #F97316 var(--pct, 50%),
            rgba(249,115,22,0.15) var(--pct, 50%),
            rgba(249,115,22,0.15) 100%
          );
          outline: none;
          cursor: pointer;
          border: none;
          padding: 0;
          margin: 0;
        }
        .bpm-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 13px;
          height: 13px;
          border-radius: 50%;
          background: #F97316;
          border: 2px solid #FEFAF3;
          box-shadow: 0 1px 4px rgba(249,115,22,0.4);
          cursor: pointer;
        }
        .bpm-slider::-moz-range-thumb {
          width: 13px;
          height: 13px;
          border-radius: 50%;
          background: #F97316;
          border: 2px solid #FEFAF3;
          box-shadow: 0 1px 4px rgba(249,115,22,0.4);
          cursor: pointer;
        }
        .bpm-slider:hover::-webkit-slider-thumb {
          background: #C2410C;
        }
      `}</style>
      <div className="bpm-root">
        <button
          className={`bpm-toggle${isMetronomeOn ? ' on' : ''}`}
          onClick={toggleMetronome}
          title={isMetronomeOn ? 'Stop metronome' : 'Start metronome'}
        >
          {isMetronomeOn ? '◼' : '▶'}
        </button>

        <input
          type="range"
          min="40"
          max="220"
          step="1"
          value={bpm}
          onChange={(e) => setBpm(Number(e.target.value))}
          className="bpm-slider"
          style={{ '--pct': `${((bpm - 40) / 180) * 100}%` } as React.CSSProperties}
          title={`BPM: ${bpm}`}
        />

        <span className="bpm-label">{bpm} BPM</span>
      </div>
    </>
  );
};
