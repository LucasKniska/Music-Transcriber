-- Run this in the Supabase SQL editor:
-- https://supabase.com/dashboard/project/kevzaugmrtwipfgbbrmf/sql/new

CREATE TABLE IF NOT EXISTS sheets (
  id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id     UUID        REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  title       TEXT        NOT NULL,
  bpm         INTEGER     NOT NULL DEFAULT 120,
  notes       JSONB       NOT NULL DEFAULT '[]',
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Row Level Security: users can only access their own sheets
ALTER TABLE sheets ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users see own sheets"
  ON sheets FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users create own sheets"
  ON sheets FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users update own sheets"
  ON sheets FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users delete own sheets"
  ON sheets FOR DELETE
  USING (auth.uid() = user_id);
