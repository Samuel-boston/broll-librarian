-- Workspace database: ~/.broll/workspaces/{id}/library.db
--
-- Every row carries workspace_id. It is redundant while each workspace has its
-- own file, but it means consolidating into one multi-tenant database later is
-- a data migration rather than a schema rewrite. Every query filters on it.

CREATE TABLE IF NOT EXISTS sources (
    id                TEXT PRIMARY KEY,
    workspace_id      TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    origin            TEXT NOT NULL CHECK (origin IN ('upload','local','drive')),
    origin_path       TEXT,
    drive_file_id     TEXT,
    drive_web_link    TEXT,
    drive_path        TEXT,
    duration_s        REAL,
    width             INTEGER,
    height            INTEGER,
    fps               REAL,
    codec             TEXT,
    filesize_bytes    INTEGER,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    indexed_at        TEXT,
    analysis_version  TEXT,
    status            TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending','analysing','indexed','failed','needs_review')),
    error_message     TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_hash ON sources (workspace_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources (workspace_id, status);
CREATE INDEX IF NOT EXISTS idx_sources_drive ON sources (workspace_id, drive_file_id);

CREATE TABLE IF NOT EXISTS shots (
    id                     TEXT PRIMARY KEY,
    workspace_id           TEXT NOT NULL,
    source_id              TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    shot_index             INTEGER NOT NULL,
    is_primary             INTEGER NOT NULL DEFAULT 0,
    start_s                REAL NOT NULL DEFAULT 0,
    end_s                  REAL NOT NULL DEFAULT 0,
    duration_s             REAL NOT NULL DEFAULT 0,
    thumbnail_path         TEXT,

    caption                TEXT,
    confidence             REAL,

    -- structured facets, one column each
    action                 TEXT,
    setting                TEXT,
    setting_detail         TEXT,
    shot_type              TEXT,
    camera_movement        TEXT,
    time_of_day            TEXT,
    colour_profile         TEXT,
    people_count           TEXT,
    has_recognisable_faces INTEGER,
    has_text_on_screen     INTEGER,
    pace                   TEXT,

    -- list facets, stored as JSON arrays and queried with json_each
    subjects_json          TEXT NOT NULL DEFAULT '[]',
    mood_json              TEXT NOT NULL DEFAULT '[]',
    usable_for_json        TEXT NOT NULL DEFAULT '[]',
    quality_flags_json     TEXT NOT NULL DEFAULT '[]',

    status                 TEXT NOT NULL DEFAULT 'pending'
                           CHECK (status IN ('pending','indexed','failed','needs_review')),
    error_message          TEXT,
    analysis_version       TEXT,
    analysed_at            TEXT,
    search_text            TEXT NOT NULL DEFAULT '',
    raw_analysis_json      TEXT,

    UNIQUE (source_id, shot_index)
);

CREATE INDEX IF NOT EXISTS idx_shots_source     ON shots (workspace_id, source_id);
CREATE INDEX IF NOT EXISTS idx_shots_status     ON shots (workspace_id, status);
CREATE INDEX IF NOT EXISTS idx_shots_action     ON shots (workspace_id, action);
CREATE INDEX IF NOT EXISTS idx_shots_setting    ON shots (workspace_id, setting);
CREATE INDEX IF NOT EXISTS idx_shots_shot_type  ON shots (workspace_id, shot_type);
CREATE INDEX IF NOT EXISTS idx_shots_movement   ON shots (workspace_id, camera_movement);
CREATE INDEX IF NOT EXISTS idx_shots_time       ON shots (workspace_id, time_of_day);
CREATE INDEX IF NOT EXISTS idx_shots_colour     ON shots (workspace_id, colour_profile);
CREATE INDEX IF NOT EXISTS idx_shots_people     ON shots (workspace_id, people_count);
CREATE INDEX IF NOT EXISTS idx_shots_pace       ON shots (workspace_id, pace);
CREATE INDEX IF NOT EXISTS idx_shots_duration   ON shots (workspace_id, duration_s);

CREATE TABLE IF NOT EXISTS tags (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'other'
                 CHECK (kind IN ('object','concept','activity','emotion','other')),
    UNIQUE (workspace_id, name)
);

CREATE TABLE IF NOT EXISTS shot_tags (
    shot_id      TEXT NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
    tag_id       INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL,
    PRIMARY KEY (shot_id, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_shot_tags_tag ON shot_tags (workspace_id, tag_id);

CREATE TABLE IF NOT EXISTS vocabulary_candidates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    field        TEXT NOT NULL,
    term         TEXT NOT NULL,
    count        INTEGER NOT NULL DEFAULT 0,
    first_seen   TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen    TEXT NOT NULL DEFAULT (datetime('now')),
    promoted     INTEGER NOT NULL DEFAULT 0,
    UNIQUE (workspace_id, field, term)
);

CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    workspace_id      TEXT NOT NULL,
    kind              TEXT NOT NULL,
    payload_json      TEXT NOT NULL DEFAULT '{}',
    status            TEXT NOT NULL DEFAULT 'queued'
                      CHECK (status IN ('queued','running','done','failed','cancelled')),
    attempts          INTEGER NOT NULL DEFAULT 0,
    last_error        TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    started_at        TEXT,
    finished_at       TEXT,
    not_before        TEXT,
    cost_estimate_usd REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (workspace_id, status, not_before);

-- FTS5 over the single denormalised shots.search_text column. search_text is
-- recomputed in exactly one place (store.recompute_search_text), which is what
-- makes the standard external-content trigger pattern safe here.
CREATE VIRTUAL TABLE IF NOT EXISTS shots_fts USING fts5(
    search_text,
    content='shots',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS shots_ai AFTER INSERT ON shots BEGIN
    INSERT INTO shots_fts(rowid, search_text) VALUES (new.rowid, new.search_text);
END;

CREATE TRIGGER IF NOT EXISTS shots_ad AFTER DELETE ON shots BEGIN
    INSERT INTO shots_fts(shots_fts, rowid, search_text) VALUES ('delete', old.rowid, old.search_text);
END;

CREATE TRIGGER IF NOT EXISTS shots_au AFTER UPDATE ON shots BEGIN
    INSERT INTO shots_fts(shots_fts, rowid, search_text) VALUES ('delete', old.rowid, old.search_text);
    INSERT INTO shots_fts(rowid, search_text) VALUES (new.rowid, new.search_text);
END;
