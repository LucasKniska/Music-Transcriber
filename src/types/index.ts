// Standard VexFlow notation codes
export type NoteDuration = 'w' | 'hd' | 'h' | 'qd' | 'q' | 'qr' | '8' | '8r' | '16';

export interface RenderedNote {
  id: string;
  keys: string[];   // e.g. ["c/4", "e/4"]
  duration: NoteDuration;
  
  // NEW FIELDS FOR BATCHING & SMART EDITING
  rawDuration: number;     // The actual performance duration in seconds
  startTimeOffset: number; // When this note started (relative to recording start)
  
  isRest: boolean;
  color?: string;   
}

export interface SessionPayload {
  title: string;
  bpm: number;
  notes: RenderedNote[];
  createdAt: string;
}

export interface Sheet {
  id: string;
  user_id: string;
  title: string;
  bpm: number;
  notes: RenderedNote[];
  created_at: string;
  updated_at: string;
}