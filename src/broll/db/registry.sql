-- Registry database: ~/.broll/registry.db
-- Holds only the list of workspaces. Each workspace has its own library.db.

CREATE TABLE IF NOT EXISTS workspaces (
    id                   TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    drive_root_folder_id TEXT,
    provider             TEXT NOT NULL DEFAULT 'gemini',
    db_path              TEXT NOT NULL,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
