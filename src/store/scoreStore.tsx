import { create } from 'zustand';
import { persist } from 'zustand/middleware'; // <--- 1. Import Middleware
import type { RenderedNote } from '../types';
import { fetchNotes, clearAllNotes, saveSession } from '../api/api'; // Import saveSession
import { quantizeDuration } from '../utils/musicMath';
import { supabase } from '../utils/supabase';

interface ActiveNoteData {
  startTime: number;
  noteName: string; 
  midi: number;
}

interface ScoreState {
  notes: RenderedNote[];
  activeNotes: Map<number, ActiveNoteData>;
  bpm: number;
  isMetronomeOn: boolean;
  currentPitch: string | null;

  setBpm: (newBpm: number) => void;
  clearScore: () => void;
  loadNotesFromBackend: () => Promise<void>;
  toggleMetronome: () => void;
  handleNoteOn: (midi: number, noteName: string) => void;
  handleNoteOff: (midi: number) => void;
  forceRenderTick: () => void;
  setCurrentPitch: (note: string | null) => void;

  // NEW ACTIONS
  saveRecording: (title: string) => Promise<string | null>;
  loadSheet: (notes: RenderedNote[], bpm: number) => void;
}

export const formatToVexKey = (note: string) => {
  if (!note) return 'c/5';
  const match = note.match(/^([a-gA-G][#b]*|rest)([0-9])$/);
  if (!match) return 'c/5';
  return `${match[1].toLowerCase()}/${match[2]}`;
};

export const useScoreStore = create<ScoreState>()(
  persist(
    (set, get) => ({
      notes: [],
      activeNotes: new Map(),
      bpm: 100,
      isMetronomeOn: false,
      currentPitch: null,

      setBpm: (newBpm) => set({ bpm: newBpm }),
      setCurrentPitch: (note) => set({ currentPitch: note }),

      handleNoteOn: (midi, noteName) => {
        const { activeNotes } = get();
        const newActive = new Map(activeNotes);
        newActive.set(midi, { startTime: Date.now() / 1000, noteName, midi });
        set({ activeNotes: newActive, currentPitch: noteName });
      },

      handleNoteOff: (midi) => {
        const { activeNotes, notes, bpm } = get();
        const noteData = activeNotes.get(midi);
        
        if (noteData) {
          const durationSec = (Date.now() / 1000) - noteData.startTime;
          const finalDuration = quantizeDuration(durationSec, bpm);
          
          // Create the note with RAW data for future editing
          const newNote: RenderedNote = {
            id: crypto.randomUUID(),
            keys: [formatToVexKey(noteData.noteName)],
            duration: finalDuration,
            
            // --- NEW DATA ---
            rawDuration: durationSec,
            startTimeOffset: noteData.startTime, // You might want to offset this by session start later
            // ----------------
            
            isRest: false,
            color: 'black'
          };

          const newActive = new Map(activeNotes);
          newActive.delete(midi);

          set({
            activeNotes: newActive,
            notes: [...notes, newNote],
            currentPitch: newActive.size > 0 ? get().currentPitch : null,
          });
          // Zustand Persist auto-saves to LocalStorage here!
        }
      },

      // --- BATCH SAVE TO SUPABASE ---
      saveRecording: async (title: string) => {
        const { notes, bpm } = get();
        if (notes.length === 0) return null;

        const createdAt = new Date().toISOString();

        // Also keep backend in sync so PDF export works
        try {
          await saveSession({ title, bpm, notes, createdAt });
        } catch (e) {
          console.warn("Backend sync failed (PDF export may not work):", e);
        }

        try {
          const { data: { user } } = await supabase.auth.getUser();
          if (!user) {
            console.warn("Not logged in — sheet not saved to database");
            return null;
          }

          const { data, error } = await supabase
            .from('sheets')
            .insert({ user_id: user.id, title, bpm, notes })
            .select('id')
            .single();

          if (error) {
            console.error("Supabase save failed:", error.message);
            return null;
          }

          console.log("Sheet saved:", data.id);
          return data.id as string;
        } catch (error) {
          console.error("Failed to save sheet:", error);
          return null;
        }
      },

      loadSheet: (notes: RenderedNote[], bpm: number) => {
        set({ notes, bpm });
      },

      forceRenderTick: () => {
        const { activeNotes } = get();
        if (activeNotes.size > 0) {
          set({ activeNotes: new Map(activeNotes) });
        }
      },

      clearScore: () => {
        set({ notes: [], activeNotes: new Map() });
        clearAllNotes().catch(e => console.error(e));
      },

      loadNotesFromBackend: async () => {
        const fetchedNotes = await fetchNotes();
        if (fetchedNotes && fetchedNotes.length > 0) {
          set({ notes: fetchedNotes });
        }
      },

      toggleMetronome: () => set((state) => ({ 
        isMetronomeOn: !state.isMetronomeOn 
      })),
    }),
    {
      name: 'maestro-backup', // Unique name for LocalStorage key
      partialize: (state) => ({ notes: state.notes, bpm: state.bpm }), // Only persist notes and settings
    }
  )
);