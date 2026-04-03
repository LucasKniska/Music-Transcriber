import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useScoreStore } from '../../store/scoreStore';
import { BTN_ACCENT_BG, BTN_ACCENT_HOVER } from '../../constants/theme';

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
  const [hovered, setHovered] = useState(false);
  const [elapsed, setElapsed] = useState(0); // seconds

  const { handleNoteOn, handleNoteOff, setCurrentPitch, bpm } = useScoreStore();

  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  // Keep a stable ref to the callback so stopAudio never needs it as a dep
  const onRecordingStoppedRef = useRef(onRecordingStopped);
  useEffect(() => { onRecordingStoppedRef.current = onRecordingStopped; }, [onRecordingStopped]);

  const stopAudio = useCallback(() => {
    // Stop timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setElapsed(0);

    // Stop media tracks (releases mic)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Close WebSocket
    if (socketRef.current) {
      socketRef.current.onclose = null; // prevent re-entrant onclose
      socketRef.current.close();
      socketRef.current = null;
    }

    setIsRecording(false);

    // Notify parent after a brief delay so last note_off events are processed
    setTimeout(() => {
      onRecordingStoppedRef.current?.();
    }, 200);
  }, []); // stable — no external deps

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

      // Start elapsed timer
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

    socket.onerror = () => {
      stopAudio();
    };

    socket.onclose = () => {
      // Server closed the connection — clean everything up
      stopAudio();
    };
  };

  // Only run cleanup on unmount
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
  const idleLabel = isMonitor ? 'Test Pitch' : 'Record Audio';
  const activeLabel = isMonitor ? 'Stop Monitor' : 'Stop Recording';
  const idleBg = isMonitor ? (hovered ? '#6d28d9' : '#7c3aed') : (hovered ? BTN_ACCENT_HOVER : BTN_ACCENT_BG);
  const activeBg = '#dc2626';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <button
        onClick={isRecording ? stopAudio : startStreaming}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        style={{
          background: isRecording ? activeBg : idleBg,
          color: 'white',
          border: isRecording ? '1px solid #b91c1c' : 'none',
          borderRadius: '0.375rem',
          padding: '0.4rem 0.875rem',
          fontSize: '0.875rem',
          fontWeight: 500,
          cursor: 'pointer',
          whiteSpace: 'nowrap',
        }}
      >
        {isRecording ? activeLabel : idleLabel}
      </button>

      {isRecording && !isMonitor && (
        <span style={{ fontSize: '0.8rem', color: '#6b7280', fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' }}>
          {formatElapsed(elapsed)} &nbsp;·&nbsp; {bpm} BPM
        </span>
      )}
    </div>
  );
};
