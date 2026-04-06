import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useScoreStore } from '../../store/scoreStore';

interface NoteEvent {
  type: 'note_on' | 'note_off' | 're_trigger' | 'volume' | 'silence_reset';
  note?: string;
  midi?: number;
  event?: string;
  value?: number;
  duration?: number;
  start_time?: number;
}

interface Props {
  onRecordingStopped?: () => void;
  mode?: 'record' | 'monitor';
}

export const RecordButton: React.FC<Props> = ({ onRecordingStopped, mode = 'record' }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const { handleNoteOn, handleNoteOff, setCurrentPitch, bpm } = useScoreStore();

  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  const onRecordingStoppedRef = useRef(onRecordingStopped);
  useEffect(() => { onRecordingStoppedRef.current = onRecordingStopped; }, [onRecordingStopped]);

  const stopAudio = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setElapsed(0);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (socketRef.current) {
      socketRef.current.onclose = null;
      socketRef.current.close();
      socketRef.current = null;
    }

    setIsRecording(false);

    setTimeout(() => {
      onRecordingStoppedRef.current?.();
    }, 200);
  }, []);

  const handleServerEvent = useCallback((data: NoteEvent) => {
    if (mode === 'monitor') {
      if (data.type === 'note_on' && data.note) setCurrentPitch(data.note);
      else if (data.type === 'note_off' || data.type === 'silence_reset') setCurrentPitch(null);
      return;
    }
    if (data.type === 'note_on' && data.midi !== undefined && data.note) {
      handleNoteOn(data.midi, data.note);
    } else if (data.type === 'note_off' && data.midi !== undefined) {
      handleNoteOff(data.midi);
    }
  }, [mode, handleNoteOn, handleNoteOff, setCurrentPitch]);

  const startStreaming = async () => {
    const socket = new WebSocket(`wss://${window.location.hostname}/ws`);
    socketRef.current = socket;

    socket.onopen = async () => {
      setIsRecording(true);
      startTimeRef.current = Date.now();

      timerRef.current = setInterval(() => {
        setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }, 1000);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            autoGainControl: false,
            noiseSuppression: false,
            channelCount: 1,
          },
        });
        streamRef.current = stream;

        const audioContext = new window.AudioContext({ sampleRate: 22050 });
        audioContextRef.current = audioContext;
        await audioContext.audioWorklet.addModule('/audioProcessor.js');

        const source = audioContext.createMediaStreamSource(stream);
        const workletNode = new AudioWorkletNode(audioContext, 'audio-processor');

        workletNode.port.onmessage = (event) => {
          if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(event.data);
          }
        };

        source.connect(workletNode);
        workletNode.connect(audioContext.destination);
      } catch (err) {
        console.error('Audio setup failed:', err);
        stopAudio();
      }
    };

    socket.onmessage = (event) => {
      try {
        const data: NoteEvent = JSON.parse(event.data);
        handleServerEvent(data);
      } catch (e) {
        console.error('JSON Parse Error', e);
      }
    };

    socket.onerror = () => { stopAudio(); };
    socket.onclose = () => { stopAudio(); };
  };

  useEffect(() => {
    return () => {
      if (socketRef.current || audioContextRef.current || streamRef.current) {
        stopAudio();
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const formatElapsed = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const isMonitor = mode === 'monitor';

  return (
    <>
      <style>{`
        @keyframes rec-pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.45); }
          50%       { box-shadow: 0 0 0 8px rgba(220,38,38,0); }
        }
        @keyframes rec-dot-blink {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.25; }
        }
        @keyframes monitor-pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(124,58,237,0.45); }
          50%       { box-shadow: 0 0 0 8px rgba(124,58,237,0); }
        }

        .rec-btn {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          border: none;
          border-radius: 999px;
          padding: 0.42rem 1.1rem 0.42rem 0.85rem;
          font-size: 0.875rem;
          font-weight: 700;
          font-family: inherit;
          cursor: pointer;
          transition: background 0.2s, transform 0.18s, box-shadow 0.2s;
          letter-spacing: 0.01em;
          white-space: nowrap;
          position: relative;
          outline: none;
        }
        .rec-btn:active { transform: scale(0.97); }

        .rec-btn-idle-record {
          background: #F97316;
          color: white;
          box-shadow: 0 2px 14px rgba(249,115,22,0.35);
        }
        .rec-btn-idle-record:hover {
          background: #C2410C;
          transform: translateY(-1px);
          box-shadow: 0 5px 20px rgba(249,115,22,0.42);
        }

        .rec-btn-idle-monitor {
          background: #7c3aed;
          color: white;
          box-shadow: 0 2px 14px rgba(124,58,237,0.35);
        }
        .rec-btn-idle-monitor:hover {
          background: #6d28d9;
          transform: translateY(-1px);
          box-shadow: 0 5px 20px rgba(124,58,237,0.42);
        }

        .rec-btn-active-record {
          background: #dc2626;
          color: white;
          box-shadow: 0 2px 14px rgba(220,38,38,0.3);
          animation: rec-pulse 1.8s ease-in-out infinite;
        }
        .rec-btn-active-record:hover {
          background: #b91c1c;
          transform: translateY(-1px);
        }

        .rec-btn-active-monitor {
          background: #6d28d9;
          color: white;
          box-shadow: 0 2px 14px rgba(109,40,217,0.3);
          animation: monitor-pulse 1.8s ease-in-out infinite;
        }
        .rec-btn-active-monitor:hover {
          background: #5b21b6;
          transform: translateY(-1px);
        }

        .rec-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: white;
          flex-shrink: 0;
          animation: rec-dot-blink 1.2s ease-in-out infinite;
        }

        .rec-mic-icon {
          flex-shrink: 0;
          opacity: 0.9;
        }

        .rec-timer {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.8rem;
          color: #92400E;
          font-variant-numeric: tabular-nums;
          white-space: nowrap;
          font-weight: 500;
        }

        .rec-timer-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #dc2626;
          animation: rec-dot-blink 1.2s ease-in-out infinite;
          flex-shrink: 0;
        }
      `}</style>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
        <button
          onClick={isRecording ? stopAudio : startStreaming}
          className={`rec-btn ${
            isRecording
              ? (isMonitor ? 'rec-btn-active-monitor' : 'rec-btn-active-record')
              : (isMonitor ? 'rec-btn-idle-monitor'  : 'rec-btn-idle-record')
          }`}
        >
          {isRecording ? (
            <span className="rec-dot" />
          ) : (
            <svg className="rec-mic-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              {isMonitor ? (
                <>
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </>
              ) : (
                <>
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </>
              )}
            </svg>
          )}
          {isRecording
            ? (isMonitor ? 'Stop Monitor' : 'Stop')
            : (isMonitor ? 'Test Pitch' : 'Record')}
        </button>

        {isRecording && !isMonitor && (
          <div className="rec-timer">
            <span className="rec-timer-dot" />
            {formatElapsed(elapsed)}
            <span style={{ opacity: 0.5 }}>·</span>
            {bpm} BPM
          </div>
        )}
      </div>
    </>
  );
};
