import React, { useEffect, useRef, useCallback } from 'react';
import { Renderer, Stave, Voice, Formatter } from 'vexflow';
import { useScoreStore, formatToVexKey } from '../../store/scoreStore';
import { convertToVexNotes } from '../../utils/VexMap';
import { quantizeDuration } from '../../utils/musicMath';
import type { RenderedNote } from '../../types';
import { NoteEditPopover } from './NoteEditPopover';

const MIN_STAVE_WIDTH = 250;
const SYSTEM_HEIGHT = 150;
const START_X = 10;
const START_Y = 20;
const BEATS_PER_MEASURE = 4;
const MEASURE_BATCH_SIZE = 4;
const NOTE_PADDING = 10;

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

  if (durationString.includes('d')) {
    value *= 1.5;
  }

  return value;
};

export const SheetMusic: React.FC = () => {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<HTMLDivElement>(null);
  const bottomAnchorRef = useRef<HTMLDivElement>(null);
  const vexNoteCache = useRef<Map<string, ReturnType<typeof convertToVexNotes>>>(new Map());
  const lastActiveQuantized = useRef<Map<number, string>>(new Map());
  const lastRenderHash = useRef<string>('');
  const noteIdMapRef = useRef<Map<string, string>>(new Map());
  const noteBoundsRef = useRef<Map<string, DOMRect>>(new Map());
  const noteStaveYRef = useRef<Map<string, number>>(new Map());

  const { notes, activeNotes, bpm, selectedNoteId, insertionPointNoteId, isModelRunning, loadNotesFromBackend, forceRenderTick, selectNote } = useScoreStore();

  useEffect(() => {
    if (notes.length === 0) {
      loadNotesFromBackend();
    }
  }, [loadNotesFromBackend]);

  useEffect(() => {
    if (activeNotes.size === 0) {
      lastActiveQuantized.current.clear();
      return;
    }
    const interval = setInterval(() => {
      const now = Date.now() / 1000;
      let changed = false;
      const current = new Map<number, string>();
      activeNotes.forEach((data, midi) => {
        const dur = quantizeDuration(now - data.startTime, bpm);
        current.set(midi, dur);
        if (lastActiveQuantized.current.get(midi) !== dur) changed = true;
      });
      if (changed || current.size !== lastActiveQuantized.current.size) {
        lastActiveQuantized.current = current;
        forceRenderTick();
      }
    }, 250);
    return () => clearInterval(interval);
  }, [activeNotes.size, forceRenderTick, bpm]);

  const handleClick = useCallback((e: React.MouseEvent) => {
    if (isModelRunning) return;
    const target = e.target as SVGElement;
    const noteGroup = target.closest('g.vf-stavenote');
    if (noteGroup) {
      const vfId = noteGroup.getAttribute('id')?.replace('vf-', '');
      const noteId = vfId ? noteIdMapRef.current.get(vfId) : null;
      if (noteId && !noteId.startsWith('temp-')) {
        selectNote(noteId);
        return;
      }
    }
    selectNote(null);
  }, [isModelRunning, selectNote]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const onScroll = () => selectNote(null);
    container.addEventListener('scroll', onScroll);
    return () => container.removeEventListener('scroll', onScroll);
  }, [selectNote]);

  useEffect(() => {
    if (!rendererRef.current || !scrollContainerRef.current) return;

    const allNotesToRender = [...notes];
    const now = Date.now() / 1000;

    activeNotes.forEach((data) => {
      const currentDurationSec = now - data.startTime;
      const liveDuration = quantizeDuration(currentDurationSec, bpm);
      allNotesToRender.push({
        id: `temp-${data.midi}`,
        keys: [formatToVexKey(data.noteName)],
        duration: liveDuration,
        rawDuration: currentDurationSec,
        startTimeOffset: data.startTime,
        isRest: false,
        color: "#F97316",
      });
    });

    const renderHash = allNotesToRender.map(n => `${n.id}:${n.duration}:${n.keys.join('+')}`).join('|') + `|sel:${selectedNoteId}`;
    if (renderHash === lastRenderHash.current) return;
    lastRenderHash.current = renderHash;

    rendererRef.current.innerHTML = '';
    const accumulatedIdMap = new Map<string, string>();
    const accumulatedStaveY = new Map<string, number>();

    const measures: RenderedNote[][] = [];
    let currentMeasure: RenderedNote[] = [];
    let currentBeats = 0;

    allNotesToRender.forEach((note) => {
      const val = getNoteDuration(note.duration);
      if (currentBeats + val > BEATS_PER_MEASURE + 0.01) {
        measures.push(currentMeasure);
        currentMeasure = [];
        currentBeats = 0;
      }
      currentMeasure.push(note);
      currentBeats += val;
    });
    if (currentMeasure.length > 0) measures.push(currentMeasure);

    const filledCount = measures.length;
    const totalStaves = Math.ceil(Math.max(filledCount, 1) / MEASURE_BATCH_SIZE) * MEASURE_BATCH_SIZE;

    const containerWidth = Math.max(800, scrollContainerRef.current.clientWidth - 40);
    const renderer = new Renderer(rendererRef.current, Renderer.Backends.SVG);
    const context = renderer.getContext();

    let x = START_X;
    let y = START_Y;

    for (let i = 0; i < totalStaves; i++) {
      const measureNotes = measures[i];
      let voice: Voice | null = null;
      let formatter: Formatter | null = null;
      let minRequiredWidth = 0;

      if (measureNotes && measureNotes.length > 0) {
        const isStable = measureNotes.every((n) => !n.id.startsWith('temp-'));
        const cacheKey = isStable ? measureNotes.map((n) => `${n.id}:${n.duration}:${n.keys.join('+')}`).join(',') + `|sel:${selectedNoteId}` : '';
        let cached = isStable ? vexNoteCache.current.get(cacheKey) : undefined;
        if (!cached) {
          cached = convertToVexNotes(measureNotes, selectedNoteId);
          if (isStable) vexNoteCache.current.set(cacheKey, cached);
        }

        const { vexNotes, idMap } = cached;
        idMap.forEach((noteId, vfId) => accumulatedIdMap.set(vfId, noteId));

        voice = new Voice({ numBeats: BEATS_PER_MEASURE, beatValue: 4 });
        voice.setStrict(false);
        voice.addTickables(vexNotes);

        formatter = new Formatter().joinVoices([voice]);
        minRequiredWidth = formatter.preCalculateMinTotalWidth([voice]);
      }

      const estimatedWidth = Math.max(MIN_STAVE_WIDTH, minRequiredWidth + NOTE_PADDING);

      if (x + estimatedWidth > containerWidth) {
        x = START_X;
        y += SYSTEM_HEIGHT;
      }

      let modifierPadding = 0;
      if (x === START_X || i === 0) modifierPadding += 30;
      if (i === 0) modifierPadding += 30;

      const finalMeasureWidth = Math.max(
        MIN_STAVE_WIDTH,
        minRequiredWidth + modifierPadding + NOTE_PADDING
      );

      const stave = new Stave(x, y, finalMeasureWidth);

      if (x === START_X || i === 0) {
        stave.addClef("treble");
        if (i === 0) stave.addTimeSignature("4/4");
      }
      stave.setContext(context).draw();

      if (voice && formatter) {
        const startX = stave.getNoteStartX();
        const endX = stave.getNoteEndX();
        const availableWidth = endX - startX - 10;

        if (availableWidth > 0) {
          formatter.format([voice], availableWidth);
          voice.draw(context, stave);
        }
      }

      if (measureNotes) {
        const topLineY = stave.getYForLine(0);
        for (const mn of measureNotes) {
          accumulatedStaveY.set(mn.id, topLineY);
        }
      }

      x += finalMeasureWidth;
    }

    const finalHeight = y + SYSTEM_HEIGHT;
    rendererRef.current.style.height = `${finalHeight}px`;
    renderer.resize(containerWidth, finalHeight);

    noteIdMapRef.current = accumulatedIdMap;
    noteStaveYRef.current = accumulatedStaveY;

    // Build bounding box map for popover positioning
    const svgEl = rendererRef.current.querySelector('svg');
    if (svgEl) {
      const bounds = new Map<string, DOMRect>();
      svgEl.querySelectorAll('g.vf-stavenote').forEach(g => {
        const vfId = g.getAttribute('id')?.replace('vf-', '');
        const noteId = vfId ? accumulatedIdMap.get(vfId) : null;
        if (noteId) {
          bounds.set(noteId, g.getBoundingClientRect());
        }
      });
      noteBoundsRef.current = bounds;
    }

    if (activeNotes.size > 0 || notes.length > 0) {
      bottomAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

  }, [notes, activeNotes, bpm, selectedNoteId]);

  const selectedBounds = selectedNoteId ? noteBoundsRef.current.get(selectedNoteId) : null;

  const insertionCursorStyle = (() => {
    if (!insertionPointNoteId || isModelRunning || !scrollContainerRef.current || !rendererRef.current) return null;
    const svgEl = rendererRef.current.querySelector('svg');
    if (!svgEl) return null;

    // rendererRef sits inside scrollContainerRef (position: relative). offsetTop/offsetLeft give
    // the offset from the container's padding edge — the same coordinate system the absolute-positioned
    // cursor uses. This keeps the cursor aligned with the SVG content regardless of container padding/border.
    const svgOffsetTop = rendererRef.current.offsetTop;
    const svgOffsetLeft = rendererRef.current.offsetLeft;

    // VexFlow defaults: 4 line-spaces above the top stave line, 5 staff lines × 10px spacing.
    // Top staff line of the first system therefore sits at START_Y + 40; staff height (top→bottom line) is 40px.
    const FIRST_STAVE_TOP_LINE_Y = START_Y + 40;
    const STAVE_HEIGHT = 40;

    if (insertionPointNoteId === '__START__') {
      return {
        top: svgOffsetTop + FIRST_STAVE_TOP_LINE_Y,
        left: svgOffsetLeft + START_X + 60,
        height: STAVE_HEIGHT,
      };
    }

    const staveTopLineY = noteStaveYRef.current.get(insertionPointNoteId);
    const ipBounds = noteBoundsRef.current.get(insertionPointNoteId);
    if (staveTopLineY === undefined || !ipBounds) return null;

    const svgRect = svgEl.getBoundingClientRect();
    const noteInternalX = ipBounds.x - svgRect.x;

    return {
      top: svgOffsetTop + staveTopLineY,
      left: svgOffsetLeft + noteInternalX + ipBounds.width + 4,
      height: STAVE_HEIGHT,
    };
  })();

  return (
    <div
      ref={scrollContainerRef}
      onClick={handleClick}
      style={{
        flex: 1,
        height: '100%',
        width: '100%',
        overflowY: 'auto',
        position: 'relative',
        background: '#FEFAF3',
        border: '1px solid rgba(249,115,22,0.1)',
        borderRadius: '0.75rem',
        padding: '0.5rem',
        cursor: isModelRunning ? 'default' : 'pointer',
      }}
    >
      <div ref={rendererRef} data-sheet-svg="true" />
      {selectedNoteId && selectedBounds && (
        <NoteEditPopover
          noteId={selectedNoteId}
          bounds={selectedBounds}
          containerRef={scrollContainerRef}
        />
      )}
      {insertionCursorStyle && (
        <div
          style={{
            position: 'absolute',
            top: insertionCursorStyle.top,
            left: insertionCursorStyle.left,
            width: 2,
            height: insertionCursorStyle.height,
            background: '#F97316',
            borderRadius: 1,
            zIndex: 50,
            animation: 'insertion-pulse 1.2s ease-in-out infinite',
          }}
        />
      )}
      <style>{`
        @keyframes insertion-pulse {
          0%, 100% { opacity: 1; box-shadow: 0 0 6px rgba(249,115,22,0.5); }
          50% { opacity: 0.4; box-shadow: 0 0 2px rgba(249,115,22,0.2); }
        }
      `}</style>
      <div ref={bottomAnchorRef} style={{ height: 1 }} />
    </div>
  );
};
