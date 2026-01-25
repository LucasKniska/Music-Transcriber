//bpmcontrol.tsx

import React from 'react';
import { useScoreStore } from '../../store/scoreStore';

export const BpmControl: React.FC = () => {
  const bpm = useScoreStore((state) => state.bpm);
  const setBpm = useScoreStore((state) => state.setBpm);
  
  const isMetronomeOn = useScoreStore((state) => state.isMetronomeOn);
  const toggleMetronome = useScoreStore((state) => state.toggleMetronome);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setBpm(Number(e.target.value));
  };

  return (
    <div className="flex items-center gap-3 px-3 py-1 border border-gray-200 rounded bg-white h-full">
      <button
        onClick={toggleMetronome}
        className={`w-8 h-8 flex items-center justify-center rounded-full transition-colors ${
          isMetronomeOn 
            ? 'bg-red-500 text-white hover:bg-red-600' 
            : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
        }`}
        title={isMetronomeOn ? "Stop Metronome" : "Start Metronome"}
      >
        {isMetronomeOn ? '◼' : '▶'}
      </button>

      <div className="flex items-center gap-2">
        <label 
          htmlFor="bpm-slider" 
          className="text-sm font-medium text-gray-700 whitespace-nowrap min-w-[80px]"
        >
          {bpm} BPM
        </label>
        
        <input
          id="bpm-slider"
          type="range"
          min="40"
          max="220"
          step="1"
          value={bpm}
          onChange={handleChange}
          className="cursor-pointer"
        />
      </div>
    </div>
  );
};