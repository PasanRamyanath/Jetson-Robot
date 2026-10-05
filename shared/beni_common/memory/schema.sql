-- Beni memory schema (§11.4). System of record: SQLite on the Jetson (/ssd/beni/memory.db).
-- Must run on SQLite 3.22 (Ubuntu 18.04): no UPSERT, no window functions, no RETURNING.
-- Every synced table has `hlc` (LWW per row) and `deleted` (tombstone; never hard-delete synced rows).
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS person (
  id TEXT PRIMARY KEY,
  display_name TEXT, aliases TEXT,          -- JSON array
  relation TEXT,                            -- owner/family/friend/guest/unknown
  card TEXT,                                -- core-memory block (<= 600 chars), rewritten by consolidation
  consent_face INTEGER DEFAULT 0, consent_voice INTEGER DEFAULT 0,
  do_not_learn INTEGER DEFAULT 0,           -- set by "forget me" (§11.8.7)
  first_seen REAL, last_seen REAL, times_seen INTEGER DEFAULT 0,
  hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS face_exemplar (
  id TEXT PRIMARY KEY, person_id TEXT,
  emb BLOB NOT NULL,                        -- fp16, L2-normalised
  quality REAL, yaw REAL, lux REAL, blur REAL, thumb_path TEXT,
  source TEXT,                              -- enroll/auto/confirmed
  created REAL, last_matched REAL, match_count INTEGER DEFAULT 0,
  hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS voice_exemplar (
  id TEXT PRIMARY KEY, person_id TEXT, emb BLOB NOT NULL,
  snr REAL, dur REAL, source TEXT, created REAL,
  hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS episode (
  id TEXT PRIMARY KEY, t_start REAL, t_end REAL,
  kind TEXT,                                -- conversation/conversation_raw/sighting/action/observation/reflection/summary
  place_id TEXT, people TEXT,               -- JSON array of person ids
  text TEXT NOT NULL,                       -- <= 500 chars
  importance REAL DEFAULT 3,                -- 1..10
  emb BLOB, keyframe_path TEXT, media_ref TEXT,   -- kind=action: JSON {key, steps, ok} (skills.py)
  parent_ids TEXT,                          -- JSON evidence pointers
  access_count INTEGER DEFAULT 0, last_access REAL,
  hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS episode_fts USING fts5(
  text, content='episode', content_rowid='rowid', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS episode_ai AFTER INSERT ON episode BEGIN
  INSERT INTO episode_fts(rowid, text) VALUES (new.rowid, new.text); END;
CREATE TRIGGER IF NOT EXISTS episode_ad AFTER DELETE ON episode BEGIN
  INSERT INTO episode_fts(episode_fts, rowid, text) VALUES ('delete', old.rowid, old.text); END;
CREATE TRIGGER IF NOT EXISTS episode_au AFTER UPDATE OF text ON episode BEGIN
  INSERT INTO episode_fts(episode_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
  INSERT INTO episode_fts(rowid, text) VALUES (new.rowid, new.text); END;

CREATE TABLE IF NOT EXISTS fact (
  id TEXT PRIMARY KEY,
  subject_id TEXT,                          -- person id, 'home', 'beni', object id
  predicate TEXT NOT NULL,
  object TEXT NOT NULL,
  text TEXT NOT NULL,
  emb BLOB,
  confidence REAL DEFAULT 0.7,
  status TEXT DEFAULT 'active',             -- active/superseded/retracted/pending_confirm
  valid_from REAL, valid_to REAL,           -- world time (bitemporal)
  recorded_at REAL,                         -- system time
  source_episode TEXT, source_kind TEXT,    -- said_by_user/inferred/observed/told_by_other
  source_person TEXT, superseded_by TEXT,
  hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS fact_subj ON fact(subject_id, predicate, status);

CREATE TABLE IF NOT EXISTS place (
  id TEXT PRIMARY KEY, name TEXT, aliases TEXT, map_id TEXT,
  x REAL, y REAL, yaw REAL, radius REAL DEFAULT 0.8, kind TEXT,
  hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS object_sighting (
  id TEXT PRIMARY KEY, label TEXT, attrs TEXT,
  x REAL, y REAL, z REAL, place_id TEXT, cam TEXT, conf REAL, t REAL, thumb_path TEXT, emb BLOB,
  hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS sighting_label ON object_sighting(label, t);
CREATE TABLE IF NOT EXISTS object_belief (
  id TEXT PRIMARY KEY, label TEXT, attrs TEXT, owner_id TEXT,
  last_place_id TEXT, last_xy TEXT, last_seen REAL,
  place_hist TEXT,                          -- JSON {place_id: count}
  hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS skill (
  id TEXT PRIMARY KEY, name TEXT, trigger TEXT, description TEXT,
  steps TEXT, preconditions TEXT, success INTEGER DEFAULT 0, fail INTEGER DEFAULT 0,
  emb BLOB, hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS routine (
  id TEXT PRIMARY KEY, person_id TEXT, activity TEXT, place_id TEXT,
  hist BLOB,                                -- 168 float32 hour-of-week bins, decayed counts
  n REAL, updated REAL, hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS feedback (
  id TEXT PRIMARY KEY, t REAL, person_id TEXT, turn_ref TEXT,
  kind TEXT,                                -- correction/praise/complaint/barge_in/explicit_rating/ignored
  prompt TEXT, response TEXT, better_response TEXT, signal REAL,
  used_for_training INTEGER DEFAULT 0, hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS nav_experience (
  id TEXT PRIMARY KEY, t REAL, from_place TEXT, to_place TEXT, success INTEGER,
  duration REAL, recoveries INTEGER, stuck_xy TEXT, notes TEXT,
  hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, hlc TEXT NOT NULL, deleted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS sync_state (peer TEXT PRIMARY KEY, last_hlc TEXT, last_sync REAL);

-- Delta extraction scans (hlc > peer.last_hlc).
CREATE INDEX IF NOT EXISTS person_hlc ON person(hlc);
CREATE INDEX IF NOT EXISTS face_exemplar_hlc ON face_exemplar(hlc);
CREATE INDEX IF NOT EXISTS voice_exemplar_hlc ON voice_exemplar(hlc);
CREATE INDEX IF NOT EXISTS episode_hlc ON episode(hlc);
CREATE INDEX IF NOT EXISTS fact_hlc ON fact(hlc);
CREATE INDEX IF NOT EXISTS place_hlc ON place(hlc);
CREATE INDEX IF NOT EXISTS object_sighting_hlc ON object_sighting(hlc);
CREATE INDEX IF NOT EXISTS object_belief_hlc ON object_belief(hlc);
CREATE INDEX IF NOT EXISTS skill_hlc ON skill(hlc);
CREATE INDEX IF NOT EXISTS routine_hlc ON routine(hlc);
CREATE INDEX IF NOT EXISTS feedback_hlc ON feedback(hlc);
CREATE INDEX IF NOT EXISTS nav_experience_hlc ON nav_experience(hlc);
CREATE INDEX IF NOT EXISTS kv_hlc ON kv(hlc);
