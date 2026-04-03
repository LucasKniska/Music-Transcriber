import React, { useEffect, useRef } from 'react';
import { Renderer, Stave, StaveNote, Voice, Formatter } from 'vexflow';
import { useScoreStore, formatToVexKey } from '../../store/scoreStore';
import { convertToVexNotes } from '../../utils/VexMap';
import { quantizeDuration } from '../../utils/musicMath'; // Keep quantize, but we will count beats locally
import type { RenderedNote } from '../../types';

const MIN_STAVE_WIDTH = 250;
const SYSTEM_HEIGHT = 150;
const START_X = 10;
const START_Y = 20;
const BEATS_PER_MEASURE = 4;
const MEASURE_BATCH_SIZE = 4;
const NOTE_PADDING = 10; // Reduced padding so we rely on formatter

// Helper: Calculate exact beat value for measure grouping
// 'q' = 1, 'h' = 2, '8' = 0.5, etc.
const getNoteDuration = (durationString: string): number => {
  const base = durationString.replace(/[rd]/g, '');
  let value = 0;

  switch (base) {
    case 'w': value = 4; break;
    case 'h': value = 2; break;
    case 'q': value = 1; break;
    case '8': value = 0.5; break;
    case '16': value = 0.25; break;
    case '32': value = 0.125; break;
    default: value = 0;
  }

  // Handle dots (multiply by 1.5)
  if (durationString.includes('d')) {
    value *= 1.5;
  }

  return value;
};

export const SheetMusic: React.FC = () => {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<HTMLDivElement>(null);
  const bottomAnchorRef = useRef<HTMLDivElement>(null);
  // Cache VexFlow StaveNote arrays for completed (non-active) measures to avoid
  // recreating expensive objects on every 100ms active-note tick.
  const vexNoteCache = useRef<Map<string, StaveNote[]>>(new Map());

  const { notes, activeNotes, bpm, loadNotesFromBackend, forceRenderTick } = useScoreStore();

  useEffect(() => {
    if (notes.length === 0) {
      loadNotesFromBackend();
    }
  }, [loadNotesFromBackend]);

  useEffect(() => {
    if (activeNotes.size === 0) return;
    const interval = setInterval(() => forceRenderTick(), 100);
    return () => clearInterval(interval);
  }, [activeNotes.size, forceRenderTick]);

  useEffect(() => {
    if (!rendererRef.current || !scrollContainerRef.current) return;

    // --- PREPARE DATA ---
    const allNotesToRender = [...notes];
    const now = Date.now() / 1000;

    activeNotes.forEach((data) => {
      const currentDurationSec = now - data.startTime;
      const liveDuration = quantizeDuration(currentDurationSec, bpm);

      allNotesToRender.push({
        id: `temp-${data.midi}`,
        keys: [formatToVexKey(data.noteName)],
        duration: liveDuration,

        // --- ADD THESE TWO LINES TO FIX TYPE ERROR ---
        rawDuration: currentDurationSec,
        startTimeOffset: data.startTime,
        // ---------------------------------------------

        isRest: false,
        color: "#ff0000"
      });
    });

    // --- RENDER ---
    rendererRef.current.innerHTML = ''; // Clear previous render

    // 1. Calculate Measures (Strict Grouping)
    const measures: RenderedNote[][] = [];
    let currentMeasure: RenderedNote[] = [];
    let currentBeats = 0;

    allNotesToRender.forEach((note) => {
      const val = getNoteDuration(note.duration);

      // Safety Check: Use 0.01 epsilon for float comparison errors
      // If adding this note pushes us over 4.01 beats, start a new measure
      if (currentBeats + val > BEATS_PER_MEASURE + 0.01) {
        measures.push(currentMeasure);
        currentMeasure = [];
        currentBeats = 0;
      }

      currentMeasure.push(note);
      currentBeats += val;
    });
    // Push the last partial measure
    if (currentMeasure.length > 0) measures.push(currentMeasure);

    // 2. Setup Renderer
    const filledCount = measures.length;
    const totalStaves = Math.ceil(Math.max(filledCount, 1) / MEASURE_BATCH_SIZE) * MEASURE_BATCH_SIZE;

    const containerWidth = Math.max(800, scrollContainerRef.current.clientWidth - 40);
    const renderer = new Renderer(rendererRef.current, Renderer.Backends.SVG);
    const context = renderer.getContext();

    let x = START_X;
    let y = START_Y;

    // 3. Render Loop
    // 3. Render Loop
    for (let i = 0; i < totalStaves; i++) {
      const measureNotes = measures[i];
      let voice: Voice | null = null;
      let formatter: Formatter | null = null;
      let minRequiredWidth = 0;

      // 1. Prepare Voice & Calculate Note Width
      if (measureNotes && measureNotes.length > 0) {
        // Use cache for fully-completed measures (no active/temp notes)
        const isStable = measureNotes.every((n) => !n.id.startsWith('temp-'));
        const cacheKey = isStable ? measureNotes.map((n) => n.id).join(',') : '';
        let vexNotes = isStable ? vexNoteCache.current.get(cacheKey) : undefined;
        if (!vexNotes) {
          vexNotes = convertToVexNotes(measureNotes);
          if (isStable) vexNoteCache.current.set(cacheKey, vexNotes);
        }
        voice = new Voice({ numBeats: BEATS_PER_MEASURE, beatValue: 4 });
        voice.setStrict(false);
        voice.addTickables(vexNotes);

        formatter = new Formatter().joinVoices([voice]);
        minRequiredWidth = formatter.preCalculateMinTotalWidth([voice]);
      }

      // 2. Check Wrap (Preview)
      // We do a rough check to see if we need to move to the next line.
      // This tells us if we will need to add a Clef (because x resets to START_X).
      const estimatedWidth = Math.max(MIN_STAVE_WIDTH, minRequiredWidth + NOTE_PADDING);

      if (x + estimatedWidth > containerWidth) {
        x = START_X;
        y += SYSTEM_HEIGHT;
      }

      // 3. Add Padding for Clef / Time Signature
      // Now that 'x' is finalized, we know if we are at the start of a line.
      let modifierPadding = 0;

      // If start of line (or first measure), we will have a Treble Clef (~30px)
      if (x === START_X || i === 0) modifierPadding += 30;

      // If first measure, we will have a Time Signature (~30px)
      if (i === 0) modifierPadding += 30;

      // 4. Calculate Final Width
      // Add the modifier padding to the Note Width so the Stave grows to accommodate both.
      const finalMeasureWidth = Math.max(
        MIN_STAVE_WIDTH,
        minRequiredWidth + modifierPadding + NOTE_PADDING
      );

      // 5. Draw Stave
      const stave = new Stave(x, y, finalMeasureWidth);

      if (x === START_X || i === 0) {
        stave.addClef("treble");
        if (i === 0) stave.addTimeSignature("4/4");
      }
      stave.setContext(context).draw();

      // 6. Format and Draw Voice
      if (voice && formatter) {
        // Now 'availableWidth' will actually be large enough for the notes
        const startX = stave.getNoteStartX();
        const endX = stave.getNoteEndX();
        const availableWidth = endX - startX - 10;

        if (availableWidth > 0) {
          formatter.format([voice], availableWidth);
          voice.draw(context, stave);
        }
      }

      x += finalMeasureWidth;
    }

    const finalHeight = y + SYSTEM_HEIGHT;
    rendererRef.current.style.height = `${finalHeight}px`;
    renderer.resize(containerWidth, finalHeight);

    if (activeNotes.size > 0 || notes.length > 0) {
      bottomAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

  }, [notes, activeNotes, bpm]);

  return (
    <div
      ref={scrollContainerRef}
      className="p-4 bg-white border rounded shadow-md overflow-y-auto relative"
      style={{ height: '400px', width: '100%' }}
    >
      <div ref={rendererRef} />
      <div ref={bottomAnchorRef} style={{ height: 1 }} />
    </div>
  );
};