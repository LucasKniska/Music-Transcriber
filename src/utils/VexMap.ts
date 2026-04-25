import { StaveNote, Dot, Accidental } from 'vexflow';
import type { RenderedNote } from '../types';

export interface VexNoteResult {
  vexNotes: StaveNote[];
  idMap: Map<string, string>; // VexFlow element id → RenderedNote id
}

export const convertToVexNotes = (
  notes: RenderedNote[],
  selectedNoteId?: string | null,
): VexNoteResult => {
  const idMap = new Map<string, string>();

  const vexNotes = notes.map((note) => {
    const baseDuration = note.duration.replace('d', '').replace('r', '');

    const staveNote = new StaveNote({
      clef: "treble",
      keys: note.keys,
      duration: baseDuration,
      autoStem: true,
    });

    const vfId = staveNote.getAttribute('id');
    idMap.set(vfId, note.id);

    note.keys.forEach((key, index) => {
      if (key.includes('#')) {
        staveNote.addModifier(new Accidental('#'), index);
      }
    });

    if (note.duration.includes('d')) {
      Dot.buildAndAttach([staveNote], { all: true });
    }

    const color = note.id === selectedNoteId ? '#2563EB' : note.color;
    if (color) {
      staveNote.setStyle({ fillStyle: color, strokeStyle: color });
    }

    return staveNote;
  });

  return { vexNotes, idMap };
};
