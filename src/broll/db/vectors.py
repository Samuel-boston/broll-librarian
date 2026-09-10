"""Vector storage for shot embeddings.

Two backends, chosen automatically:

* ``sqlite-vec`` when this Python's sqlite3 can load extensions - the intended
  implementation, using a ``vec0`` virtual table.
* an exact numpy KNN over vectors stored as blobs, when it cannot. Some Python
  builds (notably the python.org macOS framework build) compile sqlite3 without
  loadable-extension support, and a client machine is not somewhere you want to
  discover that. Exact cosine over a few tens of thousands of 384-dim vectors is
  a couple of milliseconds, so the fallback is not a compromise at this scale.

Both fix the dimension when the table is created, so changing the embedding
model requires ``broll reembed``, which drops and rebuilds the table.

This is the one module outside store.py that issues SQL; it stays inside the db
package for that reason.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def sqlite_vec_available(conn: sqlite3.Connection) -> bool:
    if not hasattr(conn, "enable_load_extension"):
        return False
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:  # pragma: no cover - depends on the interpreter
        log.debug("sqlite-vec could not be loaded: %s", exc)
        return False
    return True


class VectorIndex(ABC):
    backend: str

    def __init__(self, conn: sqlite3.Connection, workspace_id: str, dimensions: int):
        self.conn = conn
        self.workspace_id = workspace_id
        self.dimensions = dimensions

    @abstractmethod
    def create(self) -> None: ...

    @abstractmethod
    def upsert(self, shot_id: str, vector: list[float]) -> None: ...

    @abstractmethod
    def delete(self, shot_id: str) -> None: ...

    @abstractmethod
    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Return (shot_id, distance) ascending. Smaller distance is closer."""

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def get_many(self, shot_ids: list[str]) -> dict[str, list[float]]:
        """Stored vectors for these shots - used to judge how relevant a
        candidate really is, whichever side of the search found it."""

    @abstractmethod
    def drop(self) -> None: ...

    def rebuild(self, dimensions: int | None = None) -> None:
        self.drop()
        if dimensions:
            self.dimensions = dimensions
        self.create()


class SqliteVecIndex(VectorIndex):
    backend = "sqlite-vec"

    def create(self) -> None:
        self.conn.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS shot_vectors USING vec0(
                    shot_id TEXT PRIMARY KEY,
                    embedding float[{self.dimensions}]
                )"""
        )

    def upsert(self, shot_id: str, vector: list[float]) -> None:
        self.delete(shot_id)
        self.conn.execute(
            "INSERT INTO shot_vectors (shot_id, embedding) VALUES (?, ?)",
            (shot_id, _pack(vector)),
        )

    def delete(self, shot_id: str) -> None:
        self.conn.execute("DELETE FROM shot_vectors WHERE shot_id = ?", (shot_id,))

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        rows = self.conn.execute(
            "SELECT shot_id, distance FROM shot_vectors"
            " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_pack(vector), k),
        ).fetchall()
        return [(r["shot_id"], float(r["distance"])) for r in rows]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM shot_vectors").fetchone()[0])

    def get_many(self, shot_ids: list[str]) -> dict[str, list[float]]:
        if not shot_ids:
            return {}
        placeholders = ", ".join("?" for _ in shot_ids)
        rows = self.conn.execute(
            f"SELECT shot_id, embedding FROM shot_vectors WHERE shot_id IN ({placeholders})",
            tuple(shot_ids),
        ).fetchall()
        return {r["shot_id"]: _unpack(bytes(r["embedding"])) for r in rows}

    def drop(self) -> None:
        self.conn.execute("DROP TABLE IF EXISTS shot_vectors")


class NumpyVectorIndex(VectorIndex):
    """Exact cosine KNN over blobs. Vectors are stored L2-normalised."""

    backend = "numpy"

    def __init__(self, conn, workspace_id, dimensions):
        super().__init__(conn, workspace_id, dimensions)
        self._ids: list[str] | None = None
        self._matrix = None

    def create(self) -> None:
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS shot_vectors (
                   shot_id      TEXT PRIMARY KEY,
                   workspace_id TEXT NOT NULL,
                   dimensions   INTEGER NOT NULL,
                   embedding    BLOB NOT NULL
               )"""
        )

    def _invalidate(self) -> None:
        self._ids = None
        self._matrix = None

    def upsert(self, shot_id: str, vector: list[float]) -> None:
        self.conn.execute(
            """INSERT INTO shot_vectors (shot_id, workspace_id, dimensions, embedding)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (shot_id) DO UPDATE SET
                   embedding = excluded.embedding, dimensions = excluded.dimensions""",
            (shot_id, self.workspace_id, len(vector), _pack(_normalise(vector))),
        )
        self._invalidate()

    def delete(self, shot_id: str) -> None:
        self.conn.execute(
            "DELETE FROM shot_vectors WHERE workspace_id = ? AND shot_id = ?",
            (self.workspace_id, shot_id),
        )
        self._invalidate()

    def _load(self):
        import numpy as np

        if self._matrix is not None:
            return self._ids, self._matrix
        rows = self.conn.execute(
            "SELECT shot_id, embedding FROM shot_vectors WHERE workspace_id = ?",
            (self.workspace_id,),
        ).fetchall()
        self._ids = [r["shot_id"] for r in rows]
        if rows:
            self._matrix = np.frombuffer(
                b"".join(bytes(r["embedding"]) for r in rows), dtype="<f4"
            ).reshape(len(rows), -1)
        else:
            self._matrix = np.zeros((0, self.dimensions), dtype="<f4")
        return self._ids, self._matrix

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        import numpy as np

        ids, matrix = self._load()
        if not ids:
            return []
        query = np.asarray(_normalise(vector), dtype="<f4")
        if query.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"query has {query.shape[0]} dimensions but the index holds "
                f"{matrix.shape[1]}. Run `broll reembed`."
            )
        similarity = matrix @ query
        k = min(k, len(ids))
        top = np.argpartition(-similarity, k - 1)[:k]
        top = top[np.argsort(-similarity[top])]
        # Return a distance so both backends order the same way.
        return [(ids[i], float(1.0 - similarity[i])) for i in top]

    def count(self) -> int:
        return int(
            self.conn.execute(
                "SELECT COUNT(*) FROM shot_vectors WHERE workspace_id = ?",
                (self.workspace_id,),
            ).fetchone()[0]
        )

    def get_many(self, shot_ids: list[str]) -> dict[str, list[float]]:
        ids, matrix = self._load()
        position = {sid: i for i, sid in enumerate(ids)}
        return {
            sid: [float(x) for x in matrix[position[sid]]]
            for sid in shot_ids if sid in position
        }

    def drop(self) -> None:
        self.conn.execute("DROP TABLE IF EXISTS shot_vectors")
        self._invalidate()


def _normalise(vector: list[float]) -> list[float]:
    total = sum(v * v for v in vector) ** 0.5
    return [v / total for v in vector] if total else list(vector)


def get_vector_index(
    conn: sqlite3.Connection, workspace_id: str, dimensions: int
) -> VectorIndex:
    """Pick a backend, matching whatever is already on disk."""
    existing = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'shot_vectors'"
    ).fetchone()
    if existing and "vec0" in (existing["sql"] or ""):
        sqlite_vec_available(conn)  # the extension must be loaded to query it
        index: VectorIndex = SqliteVecIndex(conn, workspace_id, dimensions)
    elif existing:
        index = NumpyVectorIndex(conn, workspace_id, dimensions)
    elif sqlite_vec_available(conn):
        index = SqliteVecIndex(conn, workspace_id, dimensions)
    else:
        index = NumpyVectorIndex(conn, workspace_id, dimensions)
    index.create()
    return index
