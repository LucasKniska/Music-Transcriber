import React, { useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useScoreStore } from '../../store/scoreStore';

interface NoteEditPopoverProps {
  noteId: string;
  bounds: DOMRect;
  containerRef: React.RefObject<HTMLDivElement | null>;
}

export const NoteEditPopover: React.FC<NoteEditPopoverProps> = ({ noteId, bounds, containerRef }) => {
  const { notes, selectNote, deleteNote, shiftPitch, cycleDuration, insertNoteAfter, combineIntoChord } = useScoreStore();

  const noteIndex = useMemo(() => notes.findIndex(n => n.id === noteId), [notes, noteId]);
  const note = noteIndex !== -1 ? notes[noteIndex] : null;
  const isFirst = noteIndex === 0;
  const isLast = noteIndex === notes.length - 1;
  const isShortestDuration = note?.duration === '16';
  const isLongestDuration = note?.duration === 'w';

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      switch (e.key) {
        case 'ArrowUp': e.preventDefault(); shiftPitch(noteId, 'up'); break;
        case 'ArrowDown': e.preventDefault(); shiftPitch(noteId, 'down'); break;
        case 'ArrowLeft': e.preventDefault(); cycleDuration(noteId, 'shorter'); break;
        case 'ArrowRight': e.preventDefault(); cycleDuration(noteId, 'longer'); break;
        case 'Delete':
        case 'Backspace': e.preventDefault(); deleteNote(noteId); break;
        case 'Escape': e.preventDefault(); selectNote(null); break;
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [noteId, selectNote, deleteNote, shiftPitch, cycleDuration]);

  if (!note || !containerRef.current) return null;

  const containerRect = containerRef.current.getBoundingClientRect();
  const scrollTop = containerRef.current.scrollTop;
  const relX = bounds.x - containerRect.x + bounds.width / 2;
  const relY = bounds.y - containerRect.y + scrollTop;

  const POPOVER_WIDTH = 420;
  const POPOVER_HEIGHT = 80;
  const GAP = 12;

  const showAbove = relY > POPOVER_HEIGHT + GAP + 20;
  const top = showAbove ? relY - POPOVER_HEIGHT - GAP : relY + bounds.height + GAP;
  const left = Math.max(8, Math.min(relX - POPOVER_WIDTH / 2, containerRect.width - POPOVER_WIDTH - 8));

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: showAbove ? 6 : -6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: showAbove ? 6 : -6 }}
        transition={{ duration: 0.15 }}
        onClick={(e) => e.stopPropagation()}
        style={{
          position: 'absolute',
          top,
          left,
          zIndex: 100,
          display: 'flex',
          alignItems: 'stretch',
          gap: '6px',
          padding: '8px 10px',
          background: '#FEFAF3',
          border: '1px solid rgba(249,115,22,0.22)',
          borderRadius: '0.75rem',
          boxShadow: '0 6px 28px rgba(44,26,6,0.14)',
          fontFamily: "'Plus Jakarta Sans', system-ui, sans-serif",
        }}
      >
        {/* Pitch */}
        <BtnGroup>
          <PopoverBtn label="Up" onClick={() => shiftPitch(noteId, 'up')}
            icon={<ArrowUpIcon />} />
          <PopoverBtn label="Down" onClick={() => shiftPitch(noteId, 'down')}
            icon={<ArrowDownIcon />} />
        </BtnGroup>

        <Divider />

        {/* Duration */}
        <BtnGroup>
          <PopoverBtn label="Shorter" onClick={() => cycleDuration(noteId, 'shorter')}
            disabled={isShortestDuration} icon={<ShorterIcon />} />
          <PopoverBtn label="Longer" onClick={() => cycleDuration(noteId, 'longer')}
            disabled={isLongestDuration} icon={<LongerIcon />} />
        </BtnGroup>

        <Divider />

        {/* Delete */}
        <PopoverBtn label="Delete" onClick={() => deleteNote(noteId)}
          variant="danger" icon={<TrashIcon />} />

        <Divider />

        {/* Insert */}
        <PopoverBtn label="Add After" onClick={() => insertNoteAfter(noteId)}
          icon={<AddNoteIcon />} />

        <Divider />

        {/* Chord */}
        <BtnGroup>
          <PopoverBtn label="Chord L" onClick={() => combineIntoChord(noteId, 'left')}
            disabled={isFirst} icon={<ChordLeftIcon />} />
          <PopoverBtn label="Chord R" onClick={() => combineIntoChord(noteId, 'right')}
            disabled={isLast} icon={<ChordRightIcon />} />
        </BtnGroup>
      </motion.div>
    </AnimatePresence>
  );
};

const BtnGroup: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div style={{ display: 'flex', gap: '4px' }}>{children}</div>
);

const Divider: React.FC = () => (
  <span style={{ width: 1, alignSelf: 'stretch', margin: '4px 0', background: 'rgba(249,115,22,0.15)', flexShrink: 0 }} />
);

interface PopoverBtnProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: 'danger';
  icon: React.ReactNode;
}

const PopoverBtn: React.FC<PopoverBtnProps> = ({ label, onClick, disabled, variant, icon }) => {
  const baseColor = variant === 'danger' ? '#dc2626' : '#92400E';
  const hoverBg = variant === 'danger' ? 'rgba(220,38,38,0.08)' : 'rgba(249,115,22,0.1)';

  return (
    <button
      title={label}
      onClick={onClick}
      disabled={disabled}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '3px',
        minWidth: 42,
        height: 56,
        padding: '6px 4px',
        border: 'none',
        borderRadius: '0.4rem',
        background: 'transparent',
        color: baseColor,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.3 : 1,
        transition: 'background 0.12s',
      }}
      onMouseEnter={(e) => {
        if (!disabled) (e.currentTarget as HTMLButtonElement).style.background = hoverBg;
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
      }}
    >
      <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 22, height: 22 }}>
        {icon}
      </span>
      <span style={{ fontSize: '0.6rem', fontWeight: 600, lineHeight: 1, letterSpacing: '0.01em' }}>
        {label}
      </span>
    </button>
  );
};

/* ── SVG Icons ── */

const ArrowUpIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 13V3M3 7l5-5 5 5" />
  </svg>
);

const ArrowDownIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 3v10M3 9l5 5 5-5" />
  </svg>
);

const ShorterIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M4 8h8" />
    <path d="M6 4L2 8l4 4" />
    <path d="M10 4l4 4-4 4" />
  </svg>
);

const LongerIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M4 8h8" />
    <path d="M2 4l4 4-4 4" />
    <path d="M14 4l-4 4 4 4" />
  </svg>
);

const TrashIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2 4h12M5.33 4V2.67a1.33 1.33 0 011.34-1.34h2.66a1.33 1.33 0 011.34 1.34V4M12.67 4v9.33a1.33 1.33 0 01-1.34 1.34H4.67a1.33 1.33 0 01-1.34-1.34V4h9.34z" />
  </svg>
);

const AddNoteIcon = () => (
  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="7" cy="12" r="2.5" fill="currentColor" stroke="none" />
    <path d="M9.5 12V4" />
    <path d="M13 7v4M11 9h4" />
  </svg>
);

const ChordLeftIcon = () => (
  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M6 5l-4 4 4 4" />
    <circle cx="12" cy="7" r="2" fill="currentColor" stroke="none" />
    <circle cx="12" cy="12" r="2" fill="currentColor" stroke="none" />
    <path d="M14 12V5" />
  </svg>
);

const ChordRightIcon = () => (
  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 5l4 4-4 4" />
    <circle cx="6" cy="7" r="2" fill="currentColor" stroke="none" />
    <circle cx="6" cy="12" r="2" fill="currentColor" stroke="none" />
    <path d="M8 12V5" />
  </svg>
);
