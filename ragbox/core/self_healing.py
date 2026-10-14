"""
Self-Healing Infrastructure for RAGBox.
Provides watchdogs, content addressing, and auto-repair routines.
"""

import os
import hashlib
import asyncio
import fnmatch
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from collections import deque
import time
from dataclasses import dataclass
from loguru import logger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class HealthIssue:
    def __init__(self, issue_type: str, path: str, description: str):
        self.issue_type = issue_type
        self.path = path
        self.description = description


class ContentAddressedStorage:
    """Revision pointers and revision-safe retrieval state backed by SQLite."""

    INDEX_SCHEMA_VERSION = "5"
    COMMUNITY_SUMMARIES_KEY = "graph_community_summaries"

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "cas_state.db"
        self._cache_lock = threading.RLock()
        self._active_cache: Dict[str, str] = {}
        self._fts_enabled = False
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_hashes (
                    file_path TEXT PRIMARY KEY,
                    hash_value TEXT NOT NULL
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_revisions (
                    revision_id TEXT PRIMARY KEY,
                    graph_json TEXT NOT NULL
                )
                """
            )
            row = conn.execute(
                "SELECT value FROM state_metadata WHERE key = ?",
                ("index_schema_version",),
            ).fetchone()
            schema_changed = row is None or row[0] != self.INDEX_SCHEMA_VERSION
            if schema_changed:
                conn.execute("DROP TABLE IF EXISTS chunks_fts")

            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        doc_id UNINDEXED,
                        revision_id UNINDEXED,
                        source_path UNINDEXED,
                        start_line UNINDEXED,
                        end_line UNINDEXED,
                        content,
                        raw_content UNINDEXED,
                        tokenize='porter unicode61'
                    )
                """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError as exc:
                if "fts5" not in str(exc).lower():
                    raise
                logger.warning(
                    "SQLite was built without FTS5; sparse retrieval is disabled"
                )
            if schema_changed:
                # Schema changes alter the retrieval/graph revision contract.
                # Clearing activation pointers forces one safe rebuild while
                # legacy vectors remain hidden by active-revision filtering.
                conn.execute("DELETE FROM file_hashes")
                conn.execute("DELETE FROM graph_revisions")
                if self._fts_enabled:
                    conn.execute("DELETE FROM chunks_fts")
                conn.execute(
                    "DELETE FROM state_metadata WHERE key = ?",
                    (self.COMMUNITY_SUMMARIES_KEY,),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO state_metadata (key, value) VALUES (?, ?)",
                    ("index_schema_version", self.INDEX_SCHEMA_VERSION),
                )
            active = dict(
                conn.execute("SELECT file_path, hash_value FROM file_hashes").fetchall()
            )
        with self._cache_lock:
            self._active_cache = active

    @staticmethod
    def _path_key(file_path: str | Path) -> str:
        path = Path(file_path)
        if path.is_absolute():
            return os.path.normpath(str(path))
        return str(path.resolve(strict=False))

    def get_hash(self, file_path: Path) -> Optional[str]:
        if not file_path.exists() or not file_path.is_file():
            return None
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except OSError:
            logger.error(f"Cannot read file for hashing: {file_path}")
            return None

    def has_changed(self, file_path: Path) -> bool:
        current_hash = self.get_hash(file_path)
        if not current_hash:
            return False
        return self.get_recorded_hash(file_path) != current_hash

    def get_recorded_hash(self, file_path: str | Path) -> Optional[str]:
        """Return the active revision even when the source file was deleted."""
        with self._cache_lock:
            return self._active_cache.get(self._path_key(file_path))

    def activate(
        self,
        file_path: Path,
        hash_value: str,
        graph_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Atomically persist graph state and activate a built revision."""
        if not hash_value:
            raise ValueError("hash_value is required")
        graph_json = None
        if graph_snapshot is not None:
            if graph_snapshot.get("doc_id") != hash_value:
                raise ValueError("graph snapshot doc_id must match hash_value")
            # Serialize before entering the transaction. A malformed snapshot
            # must never advance the active revision pointer.
            graph_json = json.dumps(graph_snapshot, sort_keys=True)
        path_key = self._path_key(file_path)
        with self._cache_lock:
            with sqlite3.connect(self.db_path) as conn:
                if graph_json is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO graph_revisions "
                        "(revision_id, graph_json) VALUES (?, ?)",
                        (hash_value, graph_json),
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO file_hashes "
                    "(file_path, hash_value) VALUES (?, ?)",
                    (path_key, hash_value),
                )
            self._active_cache[path_key] = hash_value

    def update(self, file_path: Path) -> str:
        """Compatibility helper: hash and immediately activate a file."""
        new_hash = self.get_hash(file_path)
        if new_hash:
            self.activate(file_path, new_hash)
            return new_hash
        return ""

    def remove(self, file_path: Path) -> Optional[str]:
        """Remove a path and its graph snapshot when no path still uses it."""
        path_key = self._path_key(file_path)
        with self._cache_lock:
            old_hash = self._active_cache.get(path_key)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM file_hashes WHERE file_path = ?", (path_key,))
                if old_hash:
                    conn.execute(
                        """
                        DELETE FROM graph_revisions
                        WHERE revision_id = ?
                          AND NOT EXISTS (
                              SELECT 1 FROM file_hashes WHERE hash_value = ?
                          )
                        """,
                        (old_hash, old_hash),
                    )
            self._active_cache.pop(path_key, None)
        return old_hash

    def is_active(self, file_path: str | Path, revision_id: str) -> bool:
        return self.get_recorded_hash(file_path) == revision_id

    def active_revisions(self) -> Dict[str, str]:
        """Return a stable copy of all source-path activation pointers."""
        with self._cache_lock:
            return dict(self._active_cache)

    def revision_reference_count(self, revision_id: str) -> int:
        with self._cache_lock:
            return sum(value == revision_id for value in self._active_cache.values())

    def load_active_graph_snapshots(self) -> list[Dict[str, Any]]:
        """Load exactly one graph contribution for every active revision."""
        with sqlite3.connect(self.db_path) as conn:
            missing = conn.execute(
                """
                SELECT active.file_path, active.hash_value
                FROM file_hashes AS active
                LEFT JOIN graph_revisions AS graph
                  ON graph.revision_id = active.hash_value
                WHERE graph.revision_id IS NULL
                LIMIT 1
                """
            ).fetchone()
            if missing is not None:
                raise RuntimeError(
                    "Active revision has no persisted graph snapshot: "
                    f"{missing[0]} ({missing[1]})"
                )
            rows = conn.execute(
                """
                SELECT graph.revision_id, graph.graph_json
                FROM graph_revisions AS graph
                WHERE EXISTS (
                    SELECT 1 FROM file_hashes AS active
                    WHERE active.hash_value = graph.revision_id
                )
                ORDER BY graph.revision_id
                """
            ).fetchall()

        snapshots = []
        for revision_id, graph_json in rows:
            try:
                snapshot = json.loads(graph_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Corrupt graph snapshot for revision {revision_id}"
                ) from exc
            if not isinstance(snapshot, dict) or snapshot.get("doc_id") != revision_id:
                raise RuntimeError(f"Invalid graph snapshot for revision {revision_id}")
            snapshots.append(snapshot)
        return snapshots

    def save_community_summaries(self, summaries: Dict[str, str]) -> None:
        payload = json.dumps(summaries, sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO state_metadata (key, value) VALUES (?, ?)",
                (self.COMMUNITY_SUMMARIES_KEY, payload),
            )

    def load_community_summaries(self) -> Dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM state_metadata WHERE key = ?",
                (self.COMMUNITY_SUMMARIES_KEY,),
            ).fetchone()
        if row is None:
            return {}
        try:
            value = json.loads(row[0])
        except (TypeError, json.JSONDecodeError) as exc:
            logger.warning(f"Ignoring corrupt graph community summary cache: {exc}")
            return {}
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(summary, str)
            for key, summary in value.items()
        ):
            logger.warning("Ignoring invalid graph community summary cache")
            return {}
        return value

    @property
    def sparse_search_available(self) -> bool:
        return self._fts_enabled

    def index_fts_chunks(self, chunks: list[Dict[str, Any]]) -> None:
        """Stage one shadow revision in FTS; activation controls its visibility."""
        if not self._fts_enabled or not chunks:
            return
        rows = [
            (
                chunk["id"],
                chunk["metadata"].get("doc_id", ""),
                chunk["metadata"].get("revision_id", ""),
                chunk["metadata"].get("source_path", ""),
                chunk["metadata"].get("start_line"),
                chunk["metadata"].get("end_line"),
                chunk.get("sparse_content", chunk.get("content", "")),
                chunk.get("content", ""),
            )
            for chunk in chunks
        ]
        revision_ids = {row[2] for row in rows if row[2]}
        with sqlite3.connect(self.db_path) as conn:
            for revision_id in revision_ids:
                conn.execute(
                    "DELETE FROM chunks_fts WHERE revision_id = ?", (revision_id,)
                )
            conn.executemany(
                """
                INSERT INTO chunks_fts (
                    chunk_id, doc_id, revision_id, source_path,
                    start_line, end_line, content, raw_content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def delete_fts_revision(self, revision_id: str) -> None:
        if not self._fts_enabled or not revision_id:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks_fts WHERE revision_id = ?", (revision_id,))

    def delete_revision_artifacts(self, revision_id: str) -> bool:
        """Delete inactive SQLite artifacts; return False if still referenced."""
        if not revision_id:
            return False
        with self._cache_lock:
            if revision_id in self._active_cache.values():
                return False
            with sqlite3.connect(self.db_path) as conn:
                if self._fts_enabled:
                    conn.execute(
                        "DELETE FROM chunks_fts WHERE revision_id = ?", (revision_id,)
                    )
                conn.execute(
                    "DELETE FROM graph_revisions WHERE revision_id = ?", (revision_id,)
                )
        return True

    def search_fts(self, query: str, limit: int = 25) -> list[Dict[str, Any]]:
        """Return ranked FTS5 hits belonging only to active revisions."""
        if not self._fts_enabled or limit <= 0:
            return []
        terms = list(dict.fromkeys(re.findall(r"[A-Za-z0-9_]+", query.casefold())))[:16]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    f.chunk_id,
                    f.doc_id,
                    f.revision_id,
                    f.source_path,
                    f.start_line,
                    f.end_line,
                    f.raw_content,
                    bm25(chunks_fts) AS bm25_score
                FROM chunks_fts AS f
                JOIN file_hashes AS active
                  ON active.file_path = f.source_path
                 AND active.hash_value = f.revision_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [
            {
                "id": row[0],
                "content": row[6],
                "metadata": {
                    "doc_id": row[1],
                    "revision_id": row[2],
                    "source_path": row[3],
                    "start_line": int(row[4]) if row[4] is not None else None,
                    "end_line": int(row[5]) if row[5] is not None else None,
                    "retriever": "sparse",
                },
                # SQLite's FTS5 bm25 is negative; expose a larger-is-better value.
                "score": -float(row[7]),
            }
            for row in rows
        ]


@dataclass
class FileEvent:
    path: str
    event_type: str
    timestamp: float
    file_hash: str = ""


class ProductionFileWatcher(FileSystemEventHandler):
    """
    Production-grade file watcher with:
    - Debouncing (ignore rapid successive events)
    - Deduplication (SHA256-based)
    - Backpressure (bounded queue)
    - Batch processing (efficient bulk updates)
    """

    def __init__(
        self,
        index_callback: Callable[[str, Path], Any],
        loop: Optional[asyncio.AbstractEventLoop] = None,
        debounce_seconds: float = 2.0,
        max_queue_size: int = 1000,
        batch_interval: float = 0.5,
        document_dir: Optional[Path] = None,
    ):
        self.index_callback = index_callback
        self.loop = loop
        self.debounce_seconds = debounce_seconds
        self.max_queue_size = max_queue_size
        self.batch_interval = batch_interval
        self.document_dir = (
            document_dir.resolve(strict=False) if document_dir is not None else None
        )
        self._gitignore_patterns = self._load_gitignore_patterns(self.document_dir)

        self._event_timestamps: Dict[str, float] = {}
        self._event_hashes: Dict[str, str] = {}
        self._pending_queue: deque[FileEvent] = deque(maxlen=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._flush_now: Optional[asyncio.Event] = None
        self._shutdown = False
        self._state_lock = threading.Lock()

        self._events_received = 0
        self._events_deduplicated = 0
        self._events_processed = 0

    # Paths the watchdog should never index
    IGNORE_PATTERNS = {
        ".ragbox_state",
        ".git",
        "__pycache__",
        ".venv",
        "node_modules",
        ".chroma",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "coverage",
        "htmlcov",
        "poetry.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "Cargo.lock",
        "uv.lock",
    }

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if loop.is_closed():
            raise RuntimeError("Cannot bind watcher to a closed event loop")
        self.loop = loop

    @staticmethod
    def _load_gitignore_patterns(document_dir: Optional[Path]) -> tuple[str, ...]:
        if document_dir is None:
            return ()
        gitignore = document_dir / ".gitignore"
        if not gitignore.is_file():
            return ()
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            logger.warning(f"Could not read {gitignore}: {exc}")
            return ()
        return tuple(
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith(("#", "!"))
        )

    @classmethod
    def _is_ignored(
        cls,
        path: str,
        document_dir: Optional[Path] = None,
        gitignore_patterns: Optional[tuple[str, ...]] = None,
    ) -> bool:
        candidate = Path(path).resolve(strict=False)
        relative = candidate
        if document_dir is not None:
            try:
                relative = candidate.relative_to(document_dir.resolve(strict=False))
            except ValueError:
                pass
        if set(relative.parts) & cls.IGNORE_PATTERNS:
            return True

        patterns = (
            gitignore_patterns
            if gitignore_patterns is not None
            else cls._load_gitignore_patterns(document_dir)
        )
        relative_posix = relative.as_posix()
        for raw_pattern in patterns:
            directory_only = raw_pattern.endswith("/")
            pattern = raw_pattern.lstrip("/").rstrip("/")
            if not pattern:
                continue
            if "/" in pattern:
                matched = fnmatch.fnmatchcase(
                    relative_posix, pattern
                ) or fnmatch.fnmatchcase(relative_posix, f"**/{pattern}")
                if directory_only:
                    matched = matched or relative_posix.startswith(f"{pattern}/")
            else:
                matched = any(
                    fnmatch.fnmatchcase(part, pattern) for part in relative.parts
                )
            if matched:
                return True
        return False

    def _path_is_ignored(self, path: str) -> bool:
        return self._is_ignored(
            path,
            self.document_dir,
            self._gitignore_patterns,
        )

    def _queue_from_watchdog(self, event: FileEvent) -> None:
        """Cross the watchdog-thread boundary without creating a task there."""
        loop = self.loop
        if loop is None or loop.is_closed():
            logger.error(f"Watcher has no active event loop; dropping {event.path}")
            return
        loop.call_soon_threadsafe(self._accept_event, event)

    def _accept_event(self, event: FileEvent) -> None:
        """Accept an event on the bound asyncio loop and ensure one worker."""
        if self._shutdown:
            return
        with self._state_lock:
            if len(self._pending_queue) >= self.max_queue_size:
                logger.warning("Queue full, dropping oldest event")
                self._pending_queue.popleft()
            self._pending_queue.append(event)

        if self._flush_now is None:
            self._flush_now = asyncio.Event()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_batches())

    def on_modified(self, event):
        if event.is_directory or self._path_is_ignored(event.src_path):
            return
        current_time = time.time()
        file_path = event.src_path

        with self._state_lock:
            self._events_received += 1
            last_time = self._event_timestamps.get(file_path)
            if (
                last_time is not None
                and current_time - last_time < self.debounce_seconds
            ):
                self._events_deduplicated += 1
                logger.debug(f"Debounced: {file_path}")
                return

        # Compute hash for deduplication
        try:
            file_hash = self._compute_file_hash(file_path)
        except (IOError, OSError) as e:
            logger.warning(f"Cannot hash {file_path}: {e}")
            return

        with self._state_lock:
            if self._event_hashes.get(file_path) == file_hash:
                self._events_deduplicated += 1
                logger.debug(f"Content unchanged: {file_path}")
                return
            self._event_timestamps[file_path] = current_time
            self._event_hashes[file_path] = file_hash

        self._queue_from_watchdog(
            FileEvent(
                path=file_path,
                event_type="created_or_modified",
                timestamp=current_time,
                file_hash=file_hash,
            )
        )

    def on_created(self, event):
        self.on_modified(event)

    def on_deleted(self, event):
        if event.is_directory or self._path_is_ignored(event.src_path):
            return
        current_time = time.time()
        file_path = event.src_path
        with self._state_lock:
            self._events_received += 1
            self._event_timestamps.pop(file_path, None)
            self._event_hashes.pop(file_path, None)
        self._queue_from_watchdog(
            FileEvent(
                path=file_path,
                event_type="deleted",
                timestamp=current_time,
                file_hash="",
            )
        )

    def on_moved(self, event):
        if event.is_directory:
            return
        self.on_deleted(event)
        created = type(
            "CreatedEvent",
            (),
            {"is_directory": False, "src_path": event.dest_path},
        )()
        self.on_created(created)

    def _compute_file_hash(self, file_path: str) -> str:
        """Fast hash for large files (sample-based for >10MB)"""
        hasher = hashlib.sha256()

        # Get file size
        import os

        size = os.path.getsize(file_path)

        if size > 10 * 1024 * 1024:  # >10MB
            # Sample-based hash for large files
            with open(file_path, "rb") as f:
                # Read first 1MB, middle 1MB, last 1MB
                hasher.update(f.read(1024 * 1024))
                f.seek(size // 2)
                hasher.update(f.read(1024 * 1024))
                f.seek(-1024 * 1024, 2)
                hasher.update(f.read(1024 * 1024))
                hasher.update(str(size).encode())
        else:
            # Full hash for small files
            with open(file_path, "rb") as f:
                hasher.update(f.read())

        return hasher.hexdigest()[:16]  # Truncate for memory efficiency

    async def _process_batches(self) -> None:
        """Debounce, coalesce by path, and process from one async worker."""
        try:
            while not self._shutdown:
                with self._state_lock:
                    has_pending = bool(self._pending_queue)
                if not has_pending:
                    return

                assert self._flush_now is not None
                try:
                    await asyncio.wait_for(
                        self._flush_now.wait(), timeout=self.batch_interval
                    )
                except asyncio.TimeoutError:
                    pass
                self._flush_now.clear()

                with self._state_lock:
                    batch = list(self._pending_queue)
                    self._pending_queue.clear()

                # The final event for a path wins: delete→create becomes an
                # update, and create/modify→delete becomes a deletion.
                coalesced: Dict[str, FileEvent] = {}
                for pending in batch:
                    coalesced[pending.path] = pending

                logger.info(f"Processing batch of {len(coalesced)} files")
                for pending in coalesced.values():
                    try:
                        await self.index_callback(
                            pending.event_type, Path(pending.path)
                        )
                        with self._state_lock:
                            self._events_processed += 1
                    except Exception as exc:
                        logger.exception(
                            f"Failed to apply {pending.event_type} for "
                            f"{pending.path}: {exc}"
                        )
        finally:
            self._worker_task = None
            with self._state_lock:
                pending_remains = bool(self._pending_queue)
            if pending_remains and not self._shutdown:
                self._worker_task = asyncio.create_task(self._process_batches())

    async def flush(self) -> None:
        """Process all queued filesystem events before returning."""
        with self._state_lock:
            has_pending = bool(self._pending_queue)
        if has_pending and (self._worker_task is None or self._worker_task.done()):
            self._worker_task = asyncio.create_task(self._process_batches())
        if self._flush_now is not None:
            self._flush_now.set()
        while self._worker_task is not None:
            task = self._worker_task
            await asyncio.shield(task)

    def shutdown(self) -> None:
        self._shutdown = True
        loop = self.loop
        task = self._worker_task
        if loop is not None and not loop.is_closed() and task is not None:
            loop.call_soon_threadsafe(task.cancel)

    def get_stats(self) -> Dict:
        with self._state_lock:
            return {
                "events_received": self._events_received,
                "events_deduplicated": self._events_deduplicated,
                "events_processed": self._events_processed,
                "queue_size": len(self._pending_queue),
                "deduplication_rate": (
                    self._events_deduplicated / max(self._events_received, 1)
                ),
            }


class SelfHealer:
    """Detect and automatically fix issues with the RAG system."""

    def __init__(
        self,
        document_dir: Path,
        document_processor: Any,
        chunking_engine: Any,
        vector_store: Any,
        knowledge_graph: Any,
    ):
        self.document_dir = document_dir
        self.processor = document_processor
        self.chunker = chunking_engine
        self.vstore = vector_store
        self.kg = knowledge_graph
        self.cas = ContentAddressedStorage(document_dir / ".ragbox_state")
        self.observer: Any = None
        self.watcher: Optional[ProductionFileWatcher] = None
        self._watchdog_started = False
        self._event_lock: Optional[asyncio.Lock] = None
        self._building_initial = False

    def is_active_revision(self, source_path: str, revision_id: str) -> bool:
        """Return whether vector metadata points at the currently active revision."""
        if not source_path or not revision_id:
            return False
        return self.cas.is_active(source_path, revision_id)

    def _lock_for_current_loop(self) -> asyncio.Lock:
        if self._event_lock is None:
            self._event_lock = asyncio.Lock()
        return self._event_lock

    async def initial_build(self) -> None:
        """Index every changed file and fail readiness if any revision fails."""
        logger.info(f"Starting initial build for {self.document_dir}")
        failures = []
        snapshots = self.cas.load_active_graph_snapshots()
        restore_snapshots = getattr(self.kg, "restore_snapshots", None)
        if snapshots and not callable(restore_snapshots):
            raise RuntimeError(
                "Knowledge graph cannot restore persisted revision snapshots"
            )
        if callable(restore_snapshots):
            restore_snapshots(snapshots, self.cas.load_community_summaries())

        self._building_initial = True
        try:
            # Watchdog events do not exist while the process is stopped. Reconcile
            # deleted paths before scanning changed and newly-created files.
            for source_path in self.cas.active_revisions():
                path = Path(source_path)
                if not path.is_file():
                    try:
                        await self.handle_file_event("deleted", path)
                    except Exception as exc:
                        failures.append((path, exc))

            for root, dirs, files in os.walk(self.document_dir):
                dirs[:] = [
                    directory
                    for directory in dirs
                    if not directory.startswith(".")
                    and not ProductionFileWatcher._is_ignored(
                        str(Path(root) / directory), self.document_dir
                    )
                ]
                for file in files:
                    if file.startswith("."):
                        continue
                    path = Path(root) / file
                    if ProductionFileWatcher._is_ignored(str(path), self.document_dir):
                        continue
                    if self.cas.has_changed(path):
                        try:
                            await self.handle_file_event("created_or_modified", path)
                        except Exception as exc:
                            failures.append((path, exc))
        finally:
            self._building_initial = False

        await self._refresh_and_persist_graph_summaries()

        if failures:
            details = "; ".join(f"{path}: {exc}" for path, exc in failures[:5])
            raise RuntimeError(
                f"Initial index build failed for {len(failures)} file(s): {details}"
            )

    def start_watchdog(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start a watcher whose callbacks are marshalled onto one asyncio loop."""
        if self._watchdog_started:
            return
        loop = loop or asyncio.get_running_loop()
        self.watcher = ProductionFileWatcher(
            index_callback=self.handle_file_event,
            loop=loop,
            document_dir=self.document_dir,
        )
        self.observer = Observer()
        self.observer.schedule(self.watcher, str(self.document_dir), recursive=True)
        self.observer.start()
        self._watchdog_started = True
        logger.info(
            "Watchdog file watcher started with ProductionFileWatcher debouncing."
        )

    async def flush_pending(self) -> None:
        if self.watcher is None:
            return
        watcher_loop = self.watcher.loop
        current_loop = asyncio.get_running_loop()
        if watcher_loop is current_loop:
            await self.watcher.flush()
            return
        if watcher_loop is None or watcher_loop.is_closed():
            raise RuntimeError("Watcher event loop is unavailable")
        if not watcher_loop.is_running():
            self.watcher.bind_loop(current_loop)
            await self.watcher.flush()
            return
        future = asyncio.run_coroutine_threadsafe(self.watcher.flush(), watcher_loop)
        await asyncio.wrap_future(future)

    async def stop_watchdog(self) -> None:
        if self.watcher is not None:
            self.watcher.shutdown()
        if self.observer is not None and self.observer.is_alive():
            self.observer.stop()
            await asyncio.to_thread(self.observer.join, 5.0)
        self._watchdog_started = False

    async def _delete_revision(self, revision_id: str) -> int:
        if not revision_id:
            return 0
        if not self.cas.delete_revision_artifacts(revision_id):
            return 0
        return await self.vstore.delete(filter_dict={"doc_id": revision_id})

    async def _refresh_and_persist_graph_summaries(self) -> None:
        refresh = getattr(self.kg, "refresh_community_summaries", None)
        if callable(refresh):
            await refresh()
        export = getattr(self.kg, "export_community_summaries", None)
        if callable(export):
            try:
                self.cas.save_community_summaries(export())
            except (sqlite3.Error, TypeError, ValueError) as exc:
                # Summaries are derived cache data. The active revision and
                # graph snapshot remain authoritative and can regenerate them.
                logger.warning(f"Could not persist graph community summaries: {exc}")

    async def handle_file_event(self, event_type: str, path: Path) -> bool:
        """Build a shadow revision, then activate it with the graph update."""
        if path.name.startswith(".") or ProductionFileWatcher._is_ignored(
            str(path), self.document_dir
        ):
            return True

        async with self._lock_for_current_loop():
            return await self._handle_file_event_locked(event_type, path)

    async def _handle_file_event_locked(self, event_type: str, path: Path) -> bool:
        source_path = str(path.resolve(strict=False))
        old_hash = self.cas.get_recorded_hash(path)

        logger.info(f"File {event_type} detected: {path}")
        if event_type == "deleted":
            if not old_hash:
                return True
            is_last_reference = self.cas.revision_reference_count(old_hash) <= 1
            old_graph_snapshot = (
                self.kg.snapshot_document(old_hash) if is_last_reference else None
            )
            with self.kg.mutation_lock():
                try:
                    if is_last_reference:
                        self.kg.remove_document(old_hash, optimize=True)
                    self.cas.remove(path)
                except Exception:
                    if is_last_reference:
                        self.kg.rollback_document_replacement("", old_graph_snapshot)
                    raise
            removed = await self._delete_revision(old_hash) if is_last_reference else 0
            if is_last_reference and removed == 0:
                logger.warning(
                    f"No stored vectors reported for deleted revision {old_hash}"
                )
            if not self._building_initial:
                await self._refresh_and_persist_graph_summaries()
            return True

        new_hash = self.cas.get_hash(path)
        if not new_hash:
            raise RuntimeError(f"Cannot hash document: {path}")
        if old_hash == new_hash:
            return True

        try:
            document = await self.processor.process(path, new_hash)
            if not document:
                raise RuntimeError(f"No document produced for {path}")

            chunks, prepared_graph = await asyncio.gather(
                self.chunker.chunk(document),
                self.kg.prepare_document(document),
            )
            vectors_to_index = []
            for chunk in chunks:
                if "embedding" in chunk.metadata:
                    context_header = chunk.metadata.get("context_header", "")
                    sparse_content = (
                        f"{context_header}\n{chunk.content}"
                        if context_header
                        else chunk.content
                    )
                    vectors_to_index.append(
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "sparse_content": sparse_content,
                            "metadata": {
                                "doc_id": chunk.document_id,
                                "revision_id": new_hash,
                                "source_path": source_path,
                                **{
                                    k: v
                                    for k, v in chunk.metadata.items()
                                    if k != "embedding"
                                },
                            },
                            "embedding": chunk.metadata["embedding"],
                        }
                    )

            if chunks and len(vectors_to_index) != len(chunks):
                raise RuntimeError(
                    f"Embedding failed for {len(chunks) - len(vectors_to_index)} "
                    f"of {len(chunks)} chunks in {path}"
                )

            old_graph_snapshot = self.kg.snapshot_document(old_hash)

            if vectors_to_index:
                try:
                    await self.vstore.add_documents(vectors_to_index)
                    self.cas.index_fts_chunks(vectors_to_index)
                except Exception:
                    await self._delete_revision(new_hash)
                    raise
                logger.info(f"Sent {len(chunks)} chunks to Vector DB for {path.name}")

            try:
                # Queries see either the old revision or the fully-built new
                # revision. New vectors are filtered as inactive until this
                # critical section commits graph state and the CAS pointer.
                with self.kg.mutation_lock():
                    try:
                        self.kg.commit_prepared_document(
                            document,
                            prepared_graph,
                            replace_doc_id=old_hash,
                            optimize=True,
                        )
                        graph_snapshot = self.kg.snapshot_document(new_hash)
                        if graph_snapshot is None:
                            raise RuntimeError(
                                f"Knowledge graph did not produce snapshot {new_hash}"
                            )
                        self.cas.activate(path, new_hash, graph_snapshot)
                    except Exception:
                        self.kg.rollback_document_replacement(
                            new_hash, old_graph_snapshot
                        )
                        raise
            except Exception:
                await self._delete_revision(new_hash)
                raise

            if old_hash and old_hash != new_hash:
                removed = await self._delete_revision(old_hash)
                if removed == 0:
                    logger.warning(
                        f"Old revision {old_hash} is inactive but its vector "
                        "store did not report a deletion"
                    )

            if not self._building_initial:
                await self._refresh_and_persist_graph_summaries()
            return True

        except Exception as e:
            logger.error(f"Failed to auto-heal/process {path}: {e}")
            raise
