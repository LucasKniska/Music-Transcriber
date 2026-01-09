import React, { useEffect, useState, useRef } from 'react';

// Define the shape of data coming back from Python
interface NoteEvent {
  type: 'note_on' | 'note_off' | 're_trigger' | 'volume' | 'silence_reset';
  note?: string;
  midi?: number;
  event?: string;
  value?: number;
}

const AudioTranscriber: React.FC = () => {
  const [activeNotes, setActiveNotes] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [volume, setVolume] = useState(0);

  // Refs for persistent connections
  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);

  const startStreaming = async () => {
    // 1. Initialize WebSocket
    socketRef.current = new WebSocket('ws://localhost:8000');

    socketRef.current.onopen = async () => {
      console.log("WebSocket connected. Starting Audio...");
      setIsConnected(true);
      
      try {
        // 2. Request Microphone Access
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: false,
            autoGainControl: false,
            noiseSuppression: false,
            channelCount: 1
          } 
        });

        // 3. Create Audio Context at 22050Hz (Required by Basic Pitch Model)
        // This automatically handles resampling if your mic is 48k
        const audioContext = new window.AudioContext({ sampleRate: 22050 });
        audioContextRef.current = audioContext;

        await audioContext.audioWorklet.addModule('/audioProcessor.js');

        const source = audioContext.createMediaStreamSource(stream);
        const workletNode = new AudioWorkletNode(audioContext, 'audio-processor');
        workletNodeRef.current = workletNode;

        // 4. Handle Data Flow: Worklet -> Main Thread -> WebSocket
        workletNode.port.onmessage = (event) => {
          if (socketRef.current?.readyState === WebSocket.OPEN) {
            // Send Float32Array directly as binary
            socketRef.current.send(event.data);
          }
        };

        source.connect(workletNode);
        workletNode.connect(audioContext.destination); // Keep graph alive

      } catch (err) {
        console.error("Audio setup failed:", err);
        socketRef.current?.close();
      }
    };

    socketRef.current.onmessage = (event) => {
      try {
        const data: NoteEvent = JSON.parse(event.data);
        handleServerEvent(data);
      } catch (e) {
        console.error("JSON Parse Error", e);
      }
    };

    socketRef.current.onclose = () => {
      setIsConnected(false);
      stopAudio();
    };
  };

  const stopAudio = () => {
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    setActiveNotes([]);
    setVolume(0);
  };

  const handleServerEvent = (data: NoteEvent) => {
    switch (data.type) {
      case 'note_on':
        if (data.note) {
          setActiveNotes(prev => {
             // Avoid duplicate visual keys
             if (!prev.includes(data.note!)) return [...prev, data.note!];
             return prev;
          });
        }
        break;
      case 'note_off':
        setActiveNotes(prev => prev.filter(n => n !== data.note));
        break;
      case 'volume':
        if (data.value !== undefined) setVolume(data.value);
        break;
      case 'silence_reset':
        setActiveNotes([]);
        break;
    }
  };

  useEffect(() => {
    return () => stopAudio(); // Cleanup on unmount
  }, []);

  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif', textAlign: 'center' }}>
      <h1>Web Tuner</h1>
      
      {!isConnected ? (
        <button 
          onClick={startStreaming}
          style={{ padding: '15px 30px', fontSize: '18px', cursor: 'pointer', background: '#007bff', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          Start Microphone
        </button>
      ) : (
        <button 
          onClick={stopAudio}
          style={{ padding: '15px 30px', fontSize: '18px', cursor: 'pointer', background: '#dc3545', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          Stop
        </button>
      )}

      {/* Volume Bar */}
      <div style={{ margin: '30px auto', width: '300px', height: '10px', background: '#e0e0e0', borderRadius: '5px', overflow: 'hidden' }}>
        <div style={{ 
          width: `${Math.min(volume * 1000, 100)}%`, 
          height: '100%', 
          background: '#28a745',
          transition: 'width 0.1s linear'
        }} />
      </div>

      {/* Note Display */}
      <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', marginTop: '40px', minHeight: '100px' }}>
        {activeNotes.map(note => (
          <div key={note} style={{ 
            width: '80px', height: '80px', 
            background: '#ffc107', 
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '24px', fontWeight: 'bold', borderRadius: '10px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
          }}>
            {note}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AudioTranscriber;