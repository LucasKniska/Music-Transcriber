import React from 'react';
import { useScoreStore } from '../../store/scoreStore';

export const PitchIndicator: React.FC = () => {
  const currentPitch = useScoreStore((s) => s.currentPitch);

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minWidth: '90px',
        height: '36px',
        borderRadius: '0.375rem',
        background: currentPitch ? '#1f2937' : 'transparent',
        border: currentPitch ? '1px solid #374151' : '1px solid transparent',
        transition: 'background 0.12s ease, border-color 0.12s ease',
        overflow: 'hidden',
      }}
    >
      {currentPitch && (
        <span
          style={{
            color: '#f9a825',
            fontWeight: 700,
            fontSize: '1.1rem',
            letterSpacing: '0.03em',
            fontFamily: 'monospace',
          }}
        >
          ♩ {currentPitch}
        </span>
      )}
    </div>
  );
};
