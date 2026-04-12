import React, { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { supabase } from '../utils/supabase';
import { useScoreStore } from '../store/scoreStore';
import { SheetMusic } from '../components/Canvas/SheetMusic';
import type { Sheet } from '../types';
import { BTN_DANGER_COLOR, BTN_DANGER_BORDER, BTN_DANGER_HOVER } from '../constants/theme';

const SheetDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const loadSheet = useScoreStore((state) => state.loadSheet);
  const clearScore = useScoreStore((state) => state.clearScore);

  const [sheet, setSheet] = useState<Sheet | null>(null);
  const [loading, setLoading] = useState(true);
  const [title, setTitle] = useState('');
  const [editingTitle, setEditingTitle] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const titleInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!id) return;
    supabase
      .from('sheets')
      .select('*')
      .eq('id', id)
      .single()
      .then(({ data, error }) => {
        if (error || !data) {
          navigate('/dashboard');
          return;
        }
        const s = data as Sheet;
        setSheet(s);
        setTitle(s.title);
        loadSheet(s.notes, s.bpm);
        setLoading(false);
      });

    return () => {
      clearScore();
    };
  }, [id]);

  useEffect(() => {
    if (editingTitle && titleInputRef.current) {
      titleInputRef.current.focus();
      titleInputRef.current.select();
    }
  }, [editingTitle]);

  const saveTitle = async () => {
    if (!sheet || title === sheet.title) {
      setEditingTitle(false);
      return;
    }
    setSaving(true);
    const { error } = await supabase
      .from('sheets')
      .update({ title, updated_at: new Date().toISOString() })
      .eq('id', sheet.id);
    if (!error) {
      setSheet({ ...sheet, title });
    }
    setSaving(false);
    setEditingTitle(false);
  };

  const handleDelete = async () => {
    if (!sheet || !confirm(`Delete "${sheet.title}"? This cannot be undone.`)) return;
    setDeleting(true);
    await supabase.from('sheets').delete().eq('id', sheet.id);
    navigate('/dashboard');
  };

if (loading) {
    return <div style={{ padding: '2rem', color: '#6b7280' }}>Loading sheet...</div>;
  }

  if (!sheet) return null;

  return (
    <div className="container">
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flex: 1, minWidth: 0 }}>
          <Link to="/dashboard" style={{ fontSize: '0.875rem', color: '#6b7280', textDecoration: 'none', flexShrink: 0 }}>
            ← My Sheets
          </Link>
          {editingTitle ? (
            <input
              ref={titleInputRef}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onBlur={saveTitle}
              onKeyDown={(e) => { if (e.key === 'Enter') saveTitle(); if (e.key === 'Escape') { setTitle(sheet.title); setEditingTitle(false); } }}
              style={{
                fontSize: '1.25rem',
                fontWeight: 600,
                color: '#111827',
                border: '1px solid #93c5fd',
                borderRadius: '0.25rem',
                padding: '0.125rem 0.5rem',
                outline: 'none',
                minWidth: 0,
                flex: 1,
              }}
            />
          ) : (
            <h1
              onClick={() => setEditingTitle(true)}
              title="Click to edit title"
              style={{
                margin: 0,
                fontSize: '1.25rem',
                color: '#111827',
                cursor: 'text',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {saving ? title + ' (saving...)' : sheet.title}
            </h1>
          )}
        </div>
        <div className="controls">
          <span style={{ fontSize: '0.8rem', color: '#9ca3af' }}>
            {sheet.bpm} BPM · {sheet.notes.length} notes
          </span>
<button
            onClick={handleDelete}
            disabled={deleting}
            onMouseEnter={(e) => (e.currentTarget.style.background = BTN_DANGER_HOVER)}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'white')}
            style={{
              opacity: deleting ? 0.5 : 1,
              color: BTN_DANGER_COLOR,
              border: `1px solid ${BTN_DANGER_BORDER}`,
            }}
          >
            {deleting ? 'Deleting...' : 'Delete Sheet'}
          </button>
        </div>
      </header>

      <main className="main-content">
        <SheetMusic />
      </main>
    </div>
  );
};

export default SheetDetailPage;
