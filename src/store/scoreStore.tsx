import { create } from 'zustand';
import { persist } from 'zustand/middleware'; // <--- 1. Import Middleware
import type { RenderedNote } from '../types';
import { fetchNotes, clearAllNotes } from '../api/api';
import { quantizeDuration } from '../utils/musicMath';

interface ActiveNoteData {
  startTime: number;
  noteName: string;
  midi: number;
  chordGroupId: string | null;
  chordMidis: number[] | null;
}

const NOTE_OFF_DEBOUNCE_MS = 120;
const CHORD_FLUSH_MS = 150;

const noteOffDebounce: Map<number, { timerId: ReturnType<typeof setTimeout>, durationMs?: number }> = new Map();

const pendingChordNotes: Map<string, {
  notes: Array<{ midi: number, noteName: string, durationSec: number, startTime: number }>,
  timerId: ReturnType<typeof setTimeout> | null,
}> = new Map();

interface ScoreState {
  notes: RenderedNote[];
  activeNotes: Map<number, ActiveNoteData>;
  bpm: number;
  isMetronomeOn: boolean;
  currentPitch: string | null;
  isModelRunning: boolean;

  setBpm: (newBpm: number) => void;
  setModelRunning: (v: boolean) => void;
  clearScore: () => void;
  loadNotesFromBackend: () => Promise<void>;
  toggleMetronome: () => void;
  handleNoteOn: (midi: number, noteName: string, chordMidis?: number[]) => void;
  handleNoteOff: (midi: number, durationMs?: number) => void;
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
      isModelRunning: false,

      setBpm: (newBpm) => set({ bpm: newBpm }),
      setCurrentPitch: (note) => set({ currentPitch: note }),
      setModelRunning: (v) => set({ isModelRunning: v }),

      handleNoteOn: (midi, noteName, chordMidis?) => {
        const pending = noteOffDebounce.get(midi);
        if (pending) {
          clearTimeout(pending.timerId);
          noteOffDebounce.delete(midi);
          return;
        }

        const { activeNotes } = get();
        const newActive = new Map(activeNotes);

        let chordGroupId: string | null = null;
        if (chordMidis && chordMidis.length > 1) {
          for (const m of chordMidis) {
            const existing = activeNotes.get(m);
            if (existing?.chordGroupId) {
              chordGroupId = existing.chordGroupId;
              break;
            }
          }
          if (!chordGroupId) {
            chordGroupId = `chord_${Date.now()}`;
          }
        }

        newActive.set(midi, {
          startTime: Date.now() / 1000,
          noteName,
          midi,
          chordGroupId,
          chordMidis: chordMidis && chordMidis.length > 1 ? chordMidis : null,
        });
        set({ activeNotes: newActive, currentPitch: noteName });
      },

      handleNoteOff: (midi, durationMs) => {
        const { activeNotes } = get();
        const noteData = activeNotes.get(midi);
        if (!noteData) return;

        const existing = noteOffDebounce.get(midi);
        if (existing) {
          clearTimeout(existing.timerId);
        }

        const timerId = setTimeout(() => {
          noteOffDebounce.delete(midi);
          const store = get();
          const currentNoteData = store.activeNotes.get(midi);
          if (!currentNoteData) return;

          const durationSec = durationMs !== undefined
            ? durationMs / 1000
            : (Date.now() / 1000) - currentNoteData.startTime;

          const newActive = new Map(store.activeNotes);
          newActive.delete(midi);

          if (currentNoteData.chordGroupId) {
            let group = pendingChordNotes.get(currentNoteData.chordGroupId);
            if (!group) {
              group = { notes: [], timerId: null };
              pendingChordNotes.set(currentNoteData.chordGroupId, group);
            }
            group.notes.push({
              midi: currentNoteData.midi,
              noteName: currentNoteData.noteName,
              durationSec,
              startTime: currentNoteData.startTime,
            });

            if (group.timerId) clearTimeout(group.timerId);
            const groupId = currentNoteData.chordGroupId;
            group.timerId = setTimeout(() => {
              const g = pendingChordNotes.get(groupId);
              if (!g || g.notes.length === 0) {
                pendingChordNotes.delete(groupId);
                return;
              }

              const { notes: currentNotes, bpm } = get();
              const sorted = g.notes.sort((a, b) => a.midi - b.midi);
              const shortestDuration = Math.min(...sorted.map(n => n.durationSec));
              const finalDuration = quantizeDuration(shortestDuration, bpm);

              const chordNote: RenderedNote = {
                id: crypto.randomUUID(),
                keys: sorted.map(n => formatToVexKey(n.noteName)),
                duration: finalDuration,
                rawDuration: shortestDuration,
                startTimeOffset: sorted[0].startTime,
                isRest: false,
                color: 'black',
              };

              set({ notes: [...currentNotes, chordNote] });
              pendingChordNotes.delete(groupId);
            }, CHORD_FLUSH_MS);

            set({
              activeNotes: newActive,
              currentPitch: newActive.size > 0 ? store.currentPitch : null,
            });
          } else {
            const finalDuration = quantizeDuration(durationSec, store.bpm);
            const newNote: RenderedNote = {
              id: crypto.randomUUID(),
              keys: [formatToVexKey(currentNoteData.noteName)],
              duration: finalDuration,
              rawDuration: durationSec,
              startTimeOffset: currentNoteData.startTime,
              isRest: false,
              color: 'black',
            };

            set({
              activeNotes: newActive,
              notes: [...store.notes, newNote],
              currentPitch: newActive.size > 0 ? store.currentPitch : null,
            });
          }
        }, NOTE_OFF_DEBOUNCE_MS);

        noteOffDebounce.set(midi, { timerId, durationMs });
      },

      saveRecording: async (_title: string) => {
        return null;
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
        for (const { timerId } of noteOffDebounce.values()) clearTimeout(timerId);
        noteOffDebounce.clear();
        for (const group of pendingChordNotes.values()) {
          if (group.timerId) clearTimeout(group.timerId);
        }
        pendingChordNotes.clear();
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