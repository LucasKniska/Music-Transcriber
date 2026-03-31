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
}

export const RecordButton: React.FC<Props> = ({ onRecordingStopped }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [hovered, setHovered] = useState(false);

  const { handleNoteOn, handleNoteOff } = useScoreStore();

  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);

  const stopAudio = useCallback(() => {
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    setIsRecording(false);

    // Notify parent after a brief delay so last note_off events are processed
    setTimeout(() => {
      onRecordingStopped?.();
    }, 200);
  }, [onRecordingStopped]);

  const handleServerEvent = (data: NoteEvent) => {
    if (data.type === 'note_on' && data.midi !== undefined && data.note) {
      handleNoteOn(data.midi, data.note);
    } else if (data.type === 'note_off' && data.midi !== undefined) {
      handleNoteOff(data.midi);
    }
  };

  const startStreaming = async () => {
    socketRef.current = new WebSocket(`wss://${window.location.hostname}/ws`);

    socketRef.current.onopen = async () => {
      setIsRecording(true);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            autoGainControl: false,
            noiseSuppression: false,
            channelCount: 1,
          },
        });

        const audioContext = new window.AudioContext({ sampleRate: 22050 });
        audioContextRef.current = audioContext;
        await audioContext.audioWorklet.addModule('/audioProcessor.js');

        const source = audioContext.createMediaStreamSource(stream);
        const workletNode = new AudioWorkletNode(audioContext, 'audio-processor');
        workletNodeRef.current = workletNode;

        workletNode.port.onmessage = (event) => {
          if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(event.data);
          }
        };

        source.connect(workletNode);
        workletNode.connect(audioContext.destination);
      } catch (err) {
        console.error('Audio setup failed:', err);
        if (socketRef.current) {
          socketRef.current.close();
          socketRef.current = null;
        }
        setIsRecording(false);
      }
    };

    socketRef.current.onmessage = (event) => {
      try {
        const data: NoteEvent = JSON.parse(event.data);
        handleServerEvent(data);
      } catch (e) {
        console.error('JSON Parse Error', e);
      }
    };

    socketRef.current.onclose = () => {
      setIsRecording(false);
      if (audioContextRef.current) {
        stopAudio();
      }
    };
  };

  useEffect(() => {
    return () => {
      if (socketRef.current || audioContextRef.current) {
        stopAudio();
      }
    };
  }, [stopAudio]);

  return (
    <button
      onClick={isRecording ? stopAudio : startStreaming}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={
        isRecording
          ? {
              background: '#dc2626',
              color: 'white',
              border: '1px solid #b91c1c',
              borderRadius: '0.375rem',
              padding: '0.4rem 0.875rem',
              fontSize: '0.875rem',
              fontWeight: 500,
              cursor: 'pointer',
            }
          : {
              background: hovered ? BTN_ACCENT_HOVER : BTN_ACCENT_BG,
              color: 'white',
              border: 'none',
              borderRadius: '0.375rem',
              padding: '0.4rem 0.875rem',
              fontSize: '0.875rem',
              fontWeight: 500,
              cursor: 'pointer',
            }
      }
    >
      {isRecording ? 'Stop Recording' : 'Record Audio'}
    </button>
  );
};
