"""
Self-Healing Infrastructure for RAGBox.
Provides watchdogs, content addressing, and auto-repair routines.
"""
import os
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import deque
import time
from dataclasses import dataclass
from loguru import logger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent


class HealthIssue:
    def __init__(self, issue_type: str, path: str, description: str):
        self.issue_type = issue_type
        self.path = path
        self.description = description


class ContentAddressedStorage:
    """Manages document hashes to detect changes without full reprocessing. Backed by SQLite."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "cas_state.db"
        self._init_db()

    def _init_db(self) -> None:
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_hashes (
                    file_path TEXT PRIMARY KEY,
                    hash_value TEXT NOT NULL
                )
            """
            )

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

        import sqlite3

        path_str = str(file_path)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT hash_value FROM file_hashes WHERE file_path = ?", (path_str,)
            )
            row = cursor.fetchone()

        if not row or row[0] != current_hash:
            return True
        return False

    def update(self, file_path: Path) -> str:
        new_hash = self.get_hash(file_path)
        if new_hash:
            import sqlite3

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO file_hashes (file_path, hash_value) VALUES (?, ?)",
                    (str(file_path), new_hash),
                )
            return new_hash
        return ""


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
        index_callback,
        debounce_seconds: float = 2.0,
        max_queue_size: int = 1000,
        batch_interval: float = 5.0,
    ):
        self.index_callback = index_callback
        self.debounce_seconds = debounce_seconds
        self.max_queue_size = max_queue_size
        self.batch_interval = batch_interval

        # Event deduplication
        self._event_timestamps: Dict[str, float] = {}
        self._event_hashes: Dict[str, str] = {}
        self._pending_queue: deque = deque(maxlen=max_queue_size)
        self._processing = False
        self._shutdown = False

        # Statistics
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
    }

    def on_modified(self, event):
        if event.is_directory:
            return

        # Skip internal state files, VCS, and build artifacts
        path_parts = set(Path(event.src_path).parts)
        if path_parts & self.IGNORE_PATTERNS:
            return

        self._events_received += 1
        current_time = time.time()
        file_path = event.src_path

        # Debounce: Skip if same file was modified recently
        if file_path in self._event_timestamps:
            last_time = self._event_timestamps[file_path]
            if current_time - last_time < self.debounce_seconds:
                self._events_deduplicated += 1
                logger.debug(f"Debounced: {file_path}")
                return

        # Compute hash for deduplication
        try:
            file_hash = self._compute_file_hash(file_path)
        except (IOError, OSError) as e:
            logger.warning(f"Cannot hash {file_path}: {e}")
            return

        # Deduplicate: Skip if content hasn't changed
        if file_path in self._event_hashes:
            if self._event_hashes[file_path] == file_hash:
                logger.debug(f"Content unchanged: {file_path}")
                return

        self._event_timestamps[file_path] = current_time
        self._event_hashes[file_path] = file_hash

        # Add to queue with backpressure
        if len(self._pending_queue) >= self.max_queue_size:
            logger.warning("Queue full, dropping oldest event")
            self._pending_queue.popleft()

        self._pending_queue.append(
            FileEvent(
                path=file_path,
                event_type="modified",
                timestamp=current_time,
                file_hash=file_hash,
            )
        )

        # Trigger batch processing
        if not self._processing:
            asyncio.create_task(self._process_batch())

    def on_created(self, event):
        self.on_modified(event)

    def on_deleted(self, event):
        # We handle deletes differently since we can't hash them
        if event.is_directory:
            return

        self._events_received += 1
        current_time = time.time()
        file_path = event.src_path

        self._pending_queue.append(
            FileEvent(
                path=file_path,
                event_type="deleted",
                timestamp=current_time,
                file_hash="",
            )
        )

        if not self._processing:
            asyncio.create_task(self._process_batch())

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

    async def _process_batch(self):
        """Process pending events in batches with rate limiting"""
        if self._processing:
            return

        self._processing = True

        try:
            while self._pending_queue and not self._shutdown:
                # Collect batch
                batch = []
                batch_start = time.time()

                while (
                    self._pending_queue
                    and time.time() - batch_start < self.batch_interval
                ):
                    batch.append(self._pending_queue.popleft())

                if not batch:
                    break

                logger.info(f"Processing batch of {len(batch)} files")

                # Process with semaphore for concurrency control
                semaphore = asyncio.Semaphore(5)  # Max 5 concurrent indexing tasks

                async def process_with_limit(event: FileEvent):
                    async with semaphore:
                        try:
                            # Using the asyncio run_coroutine_threadsafe structure as self.index_callback is passed across loop barriers
                            future = asyncio.run_coroutine_threadsafe(
                                self.index_callback(event.event_type, Path(event.path)),
                                asyncio.get_event_loop(),
                            )
                            # Await integration
                            future.result(timeout=60)
                            self._events_processed += 1
                        except Exception as e:
                            logger.error(f"Failed to index {event.path}: {e}")

                await asyncio.gather(*[process_with_limit(e) for e in batch])

        finally:
            self._processing = False

    def get_stats(self) -> Dict:
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
        self.loop = asyncio.get_event_loop()
        self.observer = Observer()

    async def initial_build(self) -> None:
        """Scan directory and index everything that is new or changed."""
        logger.info(f"Starting initial build for {self.document_dir}")
        for root, _, files in os.walk(self.document_dir):
            for file in files:
                if file.startswith("."):
                    continue
                path = Path(root) / file
                if self.cas.has_changed(path):
                    await self.handle_file_event("created_or_modified", path)

    def start_watchdog(self) -> None:
        """Start watchdog file watcher."""
        handler = ProductionFileWatcher(index_callback=self.handle_file_event)
        self.observer.schedule(handler, str(self.document_dir), recursive=True)
        self.observer.start()
        logger.info(
            "Watchdog file watcher started with ProductionFileWatcher debouncing."
        )

    async def handle_file_event(self, event_type: str, path: Path) -> None:
        """Process a file that was added, changed, or deleted."""
        if path.name.startswith("."):
            return

        logger.info(f"File {event_type} detected: {path}")
        if event_type == "deleted":
            logger.warning(
                f"File deletion handling not fully implemented for DB delete: {path}"
            )
            return

        doc_hash = self.cas.update(path)
        if not doc_hash:
            return

        try:
            document = await self.processor.process(path, doc_hash)
            if not document:
                return

            chunks = await self.chunker.chunk(document)

            vectors_to_index = []
            for chunk in chunks:
                if "embedding" in chunk.metadata:
                    vectors_to_index.append(
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "metadata": {
                                "doc_id": chunk.document_id,
                                **{
                                    k: v
                                    for k, v in chunk.metadata.items()
                                    if k != "embedding"
                                },
                            },
                            "embedding": chunk.metadata["embedding"],
                        }
                    )

            if vectors_to_index:
                await self.vstore.add_documents(vectors_to_index)
                logger.info(f"Sent {len(chunks)} chunks to Vector DB for {path.name}")

            # Auto-Graph building
            await self.kg.build_from_documents([document])

        except Exception as e:
            logger.error(f"Failed to auto-heal/process {path}: {e}")
