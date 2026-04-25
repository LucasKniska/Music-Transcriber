import type { NoteDuration } from '../types';

/**
 * Converts raw play time into the nearest musical duration based on BPM.
 * Uses midpoints to create "buckets" for each note type.
 */
export const quantizeDuration = (seconds: number, bpm: number): NoteDuration => {
  const secondsPerBeat = 60 / bpm;
  const numBeats = seconds / secondsPerBeat;

  // We widen the '8' and 'q' buckets by pushing the triplet thresholds further away
  if (numBeats < 0.29) return '16';
  if (numBeats < 0.38) return '8r'; // Narrowed triplet bucket
  if (numBeats < 0.62) return '8';  // WIDENED 8th bucket (0.583 now falls here!)
  if (numBeats < 0.88) return 'qr'; // Narrowed triplet bucket
  if (numBeats < 1.30) return 'q';  // WIDENED quarter bucket
  if (numBeats < 1.75) return 'qd';
  if (numBeats < 2.5)  return 'h';
  if (numBeats < 3.5)  return 'hd';
  return 'w';
};

/**
 * Returns the decimal beat value for measure-filling calculations.
 */
export const getDurationValue = (duration: string): number => {
  switch (duration) {
    case 'w':  return 4;
    case 'hd': return 3;
    case 'h':  return 2;
    case 'qd': return 1.5;
    case 'q':  return 1;
    case 'qr': return 2/3;
    case '8':  return 0.5;
    case '8r': return 1/3;
    case '16': return 0.25;
    default:   return 0;
  }
};

const CHROMATIC: string[] = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b'];

export const shiftSemitone = (key: string, direction: 'up' | 'down'): string => {
  const parts = key.split('/');
  if (parts.length !== 2) return key;
  const noteName = parts[0].toLowerCase();
  let octave = parseInt(parts[1], 10);
  const idx = CHROMATIC.indexOf(noteName);
  if (idx === -1 || isNaN(octave)) return key;

  let newIdx = idx + (direction === 'up' ? 1 : -1);
  if (newIdx >= CHROMATIC.length) { newIdx = 0; octave++; }
  if (newIdx < 0) { newIdx = CHROMATIC.length - 1; octave--; }

  if (octave < 0 || octave > 8) return key;
  return `${CHROMATIC[newIdx]}/${octave}`;
};

const DURATION_ORDER: NoteDuration[] = ['16', '8', 'q', 'qd', 'h', 'hd', 'w'];

export const cycleDurationStep = (current: NoteDuration, direction: 'longer' | 'shorter'): NoteDuration => {
  let idx = DURATION_ORDER.indexOf(current);
  if (idx === -1) {
    // Triplet durations snap to nearest neighbor
    if (current === '8r') idx = direction === 'longer' ? 1 : 0;
    else if (current === 'qr') idx = direction === 'longer' ? 2 : 1;
    else return current;
  } else {
    idx += direction === 'longer' ? 1 : -1;
  }
  return DURATION_ORDER[Math.max(0, Math.min(idx, DURATION_ORDER.length - 1))];
};