import React from 'react';
import { useScoreStore } from '../../store/scoreStore';

export const PitchIndicator: React.FC = () => {
  const currentPitch = useScoreStore((s) => s.currentPitch);
  const isModelRunning = useScoreStore((s) => s.isModelRunning);

  return (
    <>
      <style>{`
        @keyframes pitch-pop {
          0%   { transform: scale(0.88); opacity: 0; }
          60%  { transform: scale(1.06); }
          100% { transform: scale(1);    opacity: 1; }
        }
        .pitch-indicator {
          display: flex;
          align-items: center;
          justify-content: center;
          min-width: 80px;
          height: 34px;
          border-radius: 999px;
          transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
          overflow: hidden;
        }
        .pitch-indicator.active {
          background: linear-gradient(135deg, #FFF0D6, #FFE4B5);
          border: 1.5px solid rgba(249,115,22,0.3);
          box-shadow: 0 2px 10px rgba(249,115,22,0.18);
        }
        .pitch-indicator.inactive {
          background: transparent;
          border: 1.5px solid transparent;
        }
        .pitch-value {
          color: #C2410C;
          font-weight: 800;
          font-size: 1rem;
          letter-spacing: 0.04em;
          font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
          animation: pitch-pop 0.18s cubic-bezier(0.22,1,0.36,1) both;
        }
      `}</style>
      <div className={`pitch-indicator ${(isModelRunning || currentPitch) ? 'active' : 'inactive'}`}>
        {currentPitch && (
          <span key={currentPitch} className="pitch-value">
            ♩ {currentPitch}
          </span>
        )}
      </div>
    </>
  );
};
