
"""
Engine Archaeologist: ScummVM Fork Analysis Tool

This sophisticated tool is designed to programmatically discover and analyze hidden game engines within ScummVM forks across various Git platforms, primarily GitHub and GitLab. It leverages advanced techniques including Abstract Syntax Tree (AST) parsing for deep code analysis, Machine Learning (ML) for engine classification, and provides both a real-time web dashboard and a robust Command-Line Interface (CLI).

Key Features:
- **Multi-Platform Scanning:** Efficiently scans forks on both GitHub and GitLab.
- **Advanced Engine Detection:** Utilizes AST parsing to identify engine-specific code patterns and structures.
- **Machine Learning Classification:** Employs a trained ML model to classify discovered engines based on their characteristics (e.g., complexity, size, code metrics).
- **Comprehensive Metrics:** Gathers detailed metrics for each engine, including lines of code (LOC), comment ratios, code complexity, and similarity scores to known ScummVM engines.
- **State Tracking & Differential Analysis:** Tracks changes in forks and engines across scans, identifying new, updated, or removed entities.
- **Concurrency Control:** Optimized for efficiency with concurrent API requests and controlled parallelism to respect rate limits.
- **Persistent Caching:** Implements ETag and local JSON caching to minimize redundant API calls and accelerate subsequent scans.
- **Interactive Web Dashboard:** Provides a real-time web interface for monitoring scan progress, viewing statistics, and visualizing engine classifications.
- **Flexible Command-Line Interface:** Offers a powerful CLI for initiating scans, training the ML model, and starting the web service.

Requirements:
- Python 3.10 or higher
- Access to GitHub and/or GitLab APIs (requires personal access tokens)
- Essential Python Libraries: `httpx`, `pendulum`, `loguru`, `rich`, `pydantic`, `pydantic-settings`, `typer`, `aiosqlite`.
- Optional Libraries for Full Functionality:
  - `tree-sitter` (for AST analysis)
  - `scikit-learn` (for ML classification)
  - `fastapi`, `uvicorn`, `python-socketio` (for web service)

Installation:
1. Clone the repository:
   `git clone https://github.com/boolforge/engine-archaeologist.git`
   `cd engine-archaeologist`
2. Install Python dependencies:
   `pip install -r requirements.txt`
3. Set up environment variables:
   Create a `.env` file in the project root with your GitHub and optional GitLab tokens:
   `GITHUB_TOKEN="your_github_personal_access_token"`
   `GITLAB_TOKEN="your_gitlab_personal_access_token"`
   (Refer to GitHub/GitLab documentation for generating personal access tokens with appropriate scopes: `repo` for GitHub, `api` for GitLab).

Usage Examples:
- **Run a standard scan:**
  `python engine-archaeologist.py scan`
- **Perform a deep scan (looks back further in time):**
  `python engine-archaeologist.py scan --deep-scan`
- **Start the web dashboard:**
  `python engine-archaeologist.py web`
- **Train the machine learning model (requires existing scan data in the database):**
  `python engine-archaeologist.py train`

For more detailed options and commands, use `python engine-archaeologist.py --help`.

Project Structure:
- `engine-archaeologist.py`: Main script containing all functionalities.
- `scummvm_discovery.db`: SQLite database for storing scan results and engine data.
- `engine_classifier_model.pkl`: (Generated) Trained machine learning model.
- `.scummvm_advanced_cache.json`: (Generated) Cache file for API responses.
- `build/tree-sitter-cpp.so`: (Generated) Tree-sitter C++ language parser.

Contribution:
Contributions are welcome! Please refer to the contribution guidelines (to be added) for more information.

License:
This project is licensed under the MIT License. See the LICENSE file for details.

"""
$ python scummvm_advanced_discovery.py web --port 8080
$ python scummvm_advanced_discovery.py train
"""

import os
import re
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import hashlib
import base64
import random
import subprocess
import pickle
import numpy as np
from urllib.parse import urlparse
from collections import defaultdict

# Core dependencies
import httpx
import pendulum
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pydantic import BaseModel, Field, conint, confloat
from pydantic_settings import BaseSettings
from typer import Typer, Option

# AST Parsing
try:
    from tree_sitter import Language, Parser
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False
    logger.warning("Tree-sitter not available. AST-based analysis skipped.")

# Web service dependencies
try:
    from fastapi import FastAPI
    from socketio import AsyncServer, ASGIApp
    from uvicorn import Config, Server
    WEB_SERVICE_AVAILABLE = True
except ImportError:
    WEB_SERVICE_AVAILABLE = False
    logger.warning("FastAPI/SocketIO not available. Web service disabled.")

# ML dependencies
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("Scikit-learn not available. ML classification disabled.")

# Database
import aiosqlite
import sqlite3

# Configuration
class Settings(BaseSettings):
    
    github_api: str = "https://api.github.com"
    gitlab_api: str = "https://gitlab.com/api/v4"
    main_owner: str = Field("scummvm", env="REPO_OWNER")
    main_repo: str = Field("scummvm", env="REPO_NAME")
    cache_file: str = ".scummvm_advanced_cache.json"
    db_file: str = "scummvm_discovery.db"
    results_file: str = "scummvm_discovery_results.json"
    user_agent: str = "ScummVM-Engine-Discovery-Tool/6.0"
    ml_model_path: str = "engine_classifier_model.pkl"
    tree_sitter_language: str = "build/tree-sitter-cpp.so"
    deep_scan_days: conint(ge=1) = Field(1095, env="DEEP_SCAN_DAYS")
    max_branches_per_fork: conint(ge=1) = Field(10, env="MAX_BRANCHES_PER_FORK")
    scan_timeout: conint(ge=60) = Field(300, env="SCAN_TIMEOUT")
    web_service_port: conint(ge=1024, le=65535) = Field(8080, env="WEB_SERVICE_PORT")
    diff_days_threshold: conint(ge=1) = Field(30, env="DIFF_DAYS_THRESHOLD")
    similarity_threshold: confloat(ge=0.0, le=1.0) = Field(0.7, env="SIMILARITY_THRESHOLD")
    max_concurrent_scans: conint(ge=1) = Field(5, env="MAX_CONCURRENT_SCANS")

    ast_queries: Dict[str, str] = {
        "engine_inheritance": """
        (class_specifier
            name: (type_identifier) @class_name
            bases: (base_clause
                (base_specifier
                    name: (qualified_identifier
                        scope: (namespace_identifier) @namespace
                        name: (identifier) @base_class
                    )
                )
            )
        )
        """,
        "engine_registration": """
        (call_expression
            function: (identifier) @func_name
            arguments: (argument_list
                (string_literal) @engine_name
            )
        )
        """,
        "plugin_registration": """
        (call_expression
            function: (identifier) @func_name
            arguments: (argument_list
                (string_literal) @plugin_name
            )
        )
        """,
        "engine_detection": """
        (struct_specifier
            name: (type_identifier) @struct_name
            (field_declaration
                (field_identifier) @field_name
            )*
        )
        """
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Initialize console
console = Console()

# --- AST Parser Setup ---
def initialize_ast_parser() -> Tuple[Optional[Parser], Dict]:
    """Initialize Tree-sitter parser for C++ code analysis."""
    if not AST_AVAILABLE:
        return None, {}
    
    Path("build").mkdir(parents=True, exist_ok=True)
    
    if not Path(settings.tree_sitter_language).exists():
        for attempt in range(3):
            try:
                if not Path("vendor/tree-sitter-cpp").exists():
                    logger.info("Cloning tree-sitter-cpp repository...")
                    subprocess.run(["git", "clone", "https://github.com/tree-sitter/tree-sitter-cpp", "vendor/tree-sitter-cpp"], check=True, timeout=300)
                
                logger.info(f"Building Tree-sitter C++ parser into {settings.tree_sitter_language}...")
                Language.build_library(settings.tree_sitter_language, [\'vendor/tree-sitter-cpp\'])
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.error(f"Tree-sitter setup failed (attempt {attempt + 1}/3): {e}")
                if attempt == 2:
                    return None, {}
    
    try:
        CPP_LANGUAGE = Language(settings.tree_sitter_language, \'cpp\')
        parser = Parser()
        parser.set_language(CPP_LANGUAGE)
        queries = {}
        for name, query_str in settings.ast_queries.items():
            try:
                queries[name] = CPP_LANGUAGE.query(query_str)
            except Exception as e:
                logger.error(f"Invalid AST query \'{name}\': {e}")
        return parser, queries
    except Exception as e:
        logger.error(f"Failed to initialize AST parser: {e}")
        return None, {}

# --- Database Models ---
class Database:
    """Database class for managing SQLite operations with async support."""
    def __init__(self, db_path: str = settings.db_file):
        self.db_path = db_path
        self._create_tables_sync()
    
    async def __aenter__(self):
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute("PRAGMA foreign_keys = ON")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()
    
    def _create_tables_sync(self):
        """Create database tables synchronously with indexes and constraints."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS scans (
                        id INTEGER PRIMARY KEY,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_forks INTEGER,
                        total_engines INTEGER,
                        duration REAL
                    );
                    
                    CREATE TABLE IF NOT EXISTS forks (
                        id INTEGER PRIMARY KEY,
                        scan_id INTEGER,
                        owner TEXT,
                        repo TEXT,
                        platform TEXT,
                        last_activity DATETIME,
                        total_score INTEGER,
                        classification TEXT,
                        stars INTEGER,
                        forks_count INTEGER,
                        open_issues INTEGER,
                        fork_url TEXT,
                        is_new BOOLEAN DEFAULT 0,
                        is_updated BOOLEAN DEFAULT 0,
                        is_removed BOOLEAN DEFAULT 0,
                        first_seen_scan_id INTEGER,
                        last_seen_scan_id INTEGER,
                        UNIQUE(owner, repo, scan_id),
                        FOREIGN KEY(scan_id) REFERENCES scans(id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS engines (
                        id INTEGER PRIMARY KEY,
                        fork_id INTEGER,
                        name TEXT,
                        path TEXT,
                        size_bytes INTEGER,
                        last_modified DATETIME,
                        complexity_score INTEGER,
                        classification TEXT,
                        branch TEXT,
                        commit_hash TEXT,
                        author TEXT,
                        commit_message TEXT,
                        is_experimental BOOLEAN,
                        is_new BOOLEAN DEFAULT 0,
                        is_updated BOOLEAN DEFAULT 0,
                        is_removed BOOLEAN DEFAULT 0,
                        first_seen_scan_id INTEGER,
                        last_seen_scan_id INTEGER,
                        validation_score INTEGER,
                        file_hash TEXT,
                        ast_signature TEXT,
                        similarity_score REAL,
                        loc INTEGER DEFAULT 0,
                        comment_lines INTEGER DEFAULT 0,
                        comment_ratio REAL DEFAULT 0.0,
                        code_complexity INTEGER DEFAULT 0,
                        UNIQUE(fork_id, name, path),
                        FOREIGN KEY(fork_id) REFERENCES forks(id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS differential (
                        id INTEGER PRIMARY KEY,
                        scan_id INTEGER,
                        new_forks INTEGER,
                        updated_forks INTEGER,
                        removed_forks INTEGER,
                        new_engines INTEGER,
                        updated_engines INTEGER,
                        removed_engines INTEGER,
                        FOREIGN KEY(scan_id) REFERENCES scans(id)
                    );
                    
                    CREATE TABLE IF NOT EXISTS engine_files (
                        engine_id INTEGER,
                        path TEXT,
                        content_hash TEXT,
                        last_modified DATETIME,
                        PRIMARY KEY (engine_id, path),
                        FOREIGN KEY(engine_id) REFERENCES engines(id)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_forks_owner_repo_scan ON forks(owner, repo, scan_id);
                    CREATE INDEX IF NOT EXISTS idx_engines_fork_name ON engines(fork_id, name);
                    CREATE INDEX IF NOT EXISTS idx_scans_timestamp ON scans(timestamp);
                """)
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error during table creation: {e}")
    
    async def save_scan(self, scan_data: Dict) -> int:



# [LLM NOTE] This is part 2 of 6 from the archive 'engine-archaeologist.py'.
# [LLM NOTE] Lines 301 to 600 are included in this part.
# [LLM NOTE] Please retain awareness of the multi-part nature of this archive.

        """Save scan metadata and return scan ID."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO scans (total_forks, total_engines, duration)
                    VALUES (?, ?, ?)
                """, (scan_data["total_forks"], scan_data["total_engines"], scan_data["duration"])))
                scan_id = cursor.lastrowid
                await self.conn.commit()
                return scan_id
        except aiosqlite.Error as e:
            logger.error(f"Error saving scan: {e}")
            return -1
    
    async def save_fork(self, scan_id: int, fork: Dict) -> int:
        """Save fork analysis with state tracking."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT id, first_seen_scan_id, last_seen_scan_id
                    FROM forks
                    WHERE owner = ? AND repo = ?
                    ORDER BY scan_id DESC LIMIT 1
                """, (fork["owner"], fork["repo"])))
                existing_fork = await cursor.fetchone()
                
                if existing_fork:
                    fork_id, first_seen, last_seen = existing_fork
                    is_new = False
                    is_updated = await self.is_fork_updated(fork_id, fork)
                    is_removed = False
                    new_last_seen = scan_id
                else:
                    is_new = True
                    is_updated = False
                    is_removed = False
                    first_seen = scan_id
                    new_last_seen = scan_id
                
                if existing_fork:
                    await cursor.execute("""
                        UPDATE forks
                        SET scan_id = ?, last_activity = ?, total_score = ?, classification = ?,
                            stars = ?, forks_count = ?, open_issues = ?, fork_url = ?,
                            is_new = ?, is_updated = ?, is_removed = ?, last_seen_scan_id = ?
                        WHERE id = ?
                    """, (
                        scan_id, fork["last_activity"], fork["total_score"], fork["classification"],
                        fork["stars"], fork["forks"], fork["open_issues"], fork["fork_url"],
                        int(is_new), int(is_updated), int(is_removed), new_last_seen, fork_id
                    ))
                else:
                    await cursor.execute("""
                        INSERT INTO forks (
                            scan_id, owner, repo, platform, last_activity, total_score, classification,
                            stars, forks_count, open_issues, fork_url, is_new, is_updated, is_removed,
                            first_seen_scan_id, last_seen_scan_id
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        scan_id, fork["owner"], fork["repo"], fork["platform"], fork["last_activity"],
                        fork["total_score"], fork["classification"], fork["stars"], fork["forks"],
                        fork["open_issues"], fork["fork_url"], int(is_new), int(is_updated),
                        int(is_removed), first_seen, new_last_seen
                    ))
                    fork_id = cursor.lastrowid
                
                await self.conn.commit()
                return fork_id
        except aiosqlite.Error as e:
            logger.error(f"Error saving fork: {e}")
            return -1
    
    async def is_fork_updated(self, fork_id: int, new_fork: Dict) -> bool:
        """Check if a fork has significant updates."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT total_score, last_activity, classification
                    FROM forks
                    WHERE id = ?
                """, (fork_id,)))
                old_fork = await cursor.fetchone()
                
                if not old_fork:
                    return True
                
                old_score, old_activity, old_class = old_fork
                new_score = new_fork.get("total_score", 0)
                new_activity = new_fork.get("last_activity")
                new_class = new_fork.get("classification", "")
                
                if old_score > 0 and abs(new_score - old_score) / old_score > 0.1:
                    return True
                if new_class != old_class:
                    return True
                if new_activity and old_activity:
                    if (pendulum.parse(new_activity) - pendulum.parse(old_activity)).in_days() > settings.diff_days_threshold:
                        return True
                return False
        except aiosqlite.Error as e:
            logger.error(f"Error checking fork update: {e}")
            return True
    
    async def save_engine(self, fork_id: int, engine: Dict) -> int:
        """Save engine information with state tracking."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT id, first_seen_scan_id, last_seen_scan_id, file_hash
                    FROM engines
                    WHERE fork_id = ? AND name = ? AND path = ?
                """, (fork_id, engine["name"], engine["path"])))
                existing_engine = await cursor.fetchone()
                
                if existing_engine:
                    engine_id, first_seen, last_seen, old_file_hash = existing_engine
                    is_new = False
                    is_updated = await self.is_engine_updated(engine_id, engine)
                    is_removed = False
                    new_last_seen = last_seen
                else:
                    is_new = True
                    is_updated = False
                    is_removed = False
                    first_seen = None
                    new_last_seen = None
                
                file_hash = self.calculate_engine_hash(engine)
                
                if existing_engine:
                    await cursor.execute("""
                        UPDATE engines
                        SET size_bytes = ?, last_modified = ?, complexity_score = ?, classification = ?,
                            branch = ?, commit_hash = ?, author = ?, commit_message = ?, is_experimental = ?,
                            is_new = ?, is_updated = ?, is_removed = ?, validation_score = ?, file_hash = ?,
                            ast_signature = ?, similarity_score = ?, loc = ?, comment_lines = ?,
                            comment_ratio = ?, code_complexity = ?
                        WHERE id = ?
                    """, (
                        engine["size_bytes"], engine["last_modified"], engine["complexity_score"],
                        engine["classification"], engine["branch"], engine["commit_hash"], engine["author"],
                        engine["commit_message"], int(engine["is_experimental"]), int(is_new), int(is_updated),
                        int(is_removed), engine["validation_score"], file_hash, engine["ast_signature"],
                        engine["similarity_score"], engine["loc"], engine["comment_lines"],
                        engine["comment_ratio"], engine["code_complexity"], engine_id
                    ))
                else:
                    await cursor.execute("""
                        INSERT INTO engines (
                            fork_id, name, path, size_bytes, last_modified, complexity_score, classification,
                            branch, commit_hash, author, commit_message, is_experimental, is_new, is_updated,
                            is_removed, validation_score, file_hash, ast_signature, similarity_score,
                            loc, comment_lines, comment_ratio, code_complexity
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fork_id, engine["name"], engine["path"], engine["size_bytes"], engine["last_modified"],
                        engine["complexity_score"], engine["classification"], engine["branch"], engine["commit_hash"],
                        engine["author"], engine["commit_message"], int(engine["is_experimental"]), int(is_new),
                        int(is_updated), int(is_removed), engine["validation_score"], file_hash,
                        engine["ast_signature"], engine["similarity_score"], engine["loc"], engine["comment_lines"],
                        engine["comment_ratio"], engine["code_complexity"]
                    ))
                    engine_id = cursor.lastrowid
                
                for file_path in engine.get("files", []):
                    await self.save_engine_file(engine_id, file_path)
                
                await self.conn.commit()
                return engine_id
        except aiosqlite.Error as e:
            logger.error(f"Error saving engine: {e}")
            return -1
    
    async def is_engine_updated(self, engine_id: int, new_engine: Dict) -> bool:
        """Check if an engine has significant updates."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT complexity_score, validation_score, file_hash, last_modified
                    FROM engines
                    WHERE id = ?
                """, (engine_id,)))
                old_engine = await cursor.fetchone()
                
                if not old_engine:
                    return True
                
                old_score, old_validation, old_hash, old_modified = old_engine
                new_score = new_engine.get("complexity_score", 0)
                new_validation = new_engine.get("validation_score", 0)
                new_hash = self.calculate_engine_hash(new_engine)
                new_modified = new_engine.get("last_modified")
                
                if new_hash != old_hash:
                    return True
                if abs(new_score - old_score) > 10:
                    return True
                if abs(new_validation - old_validation) > 15:
                    return True
                if new_modified and old_modified:
                    if (pendulum.parse(new_modified) - pendulum.parse(old_modified)).in_days() > settings.diff_days_threshold:
                        return True
                return False
        except aiosqlite.Error as e:
            logger.error(f"Error checking engine update: {e}")
            return True
    
    def calculate_engine_hash(self, engine: Dict) -> str:
        """Calculate a hash based on engine file contents and paths."""
        file_data = "\n".join(sorted(f"{f}:{engine.get("content_" + f, "")}" for f in engine.get("files", [])))
        return hashlib.sha256(file_data.encode()).hexdigest()
    
    async def save_engine_file(self, engine_id: int, file_path: str):
        """Save engine file information."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT OR REPLACE INTO engine_files (engine_id, path)
                    VALUES (?, ?)
                """, (engine_id, file_path)))
                await self.conn.commit()
        except aiosqlite.Error as e:
            logger.error(f"Error saving engine file: {e}")
    
    async def save_differential(self, scan_id: int, diff_data: Dict):
        """Save differential analysis results."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO differential (
                        scan_id, new_forks, updated_forks, removed_forks,
                        new_engines, updated_engines, removed_engines
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    scan_id, diff_data["new_forks"], diff_data["updated_forks"], diff_data["removed_forks"],
                    diff_data["new_engines"], diff_data["updated_engines"], diff_data["removed_engines"]
                )))
                await self.conn.commit()
        except aiosqlite.Error as e:
            logger.error(f"Error saving differential data: {e}")
    
    async def mark_removed_entities(self, current_scan_id: int):
        """Mark forks and engines not seen in the current scan."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    UPDATE forks
                    SET is_removed = 1, last_seen_scan_id = ?
                    WHERE last_seen_scan_id < ? AND is_removed = 0
                """, (current_scan_id - 1, current_scan_id)))
                
                await cursor.execute("""
                    UPDATE engines
                    SET is_removed = 1, last_seen_scan_id = ?
                    WHERE last_seen_scan_id < ? AND is_removed = 0
                """, (current_scan_id - 1, current_scan_id)))
                
                await self.conn.commit()
        except aiosqlite.Error as e:
            logger.error(f"Error marking removed entities: {e}")
    
    async def get_last_scan(self):
        """Get the last scan from the database."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM scans ORDER BY timestamp DESC LIMIT 1")
                return await cursor.fetchone()
        except aiosqlite.Error as e:
            logger.error(f"Error getting last scan: {e}")
            return None
    
    async def get_forks_for_scan(self, scan_id: int):
        """Get all forks for a specific scan."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM forks WHERE scan_id = ?", (scan_id,)))
                return await cursor.fetchall()
        except aiosqlite.Error as e:
            logger.error(f"Error getting forks: {e}")
            return []
    
    async def get_engines_for_fork(self, fork_id: int):
        """Get all engines for a specific fork."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM engines WHERE fork_id = ?", (fork_id,)))
                return await cursor.fetchall()
        except aiosqlite.Error as e:
            logger.error(f"Error getting engines: {e}")
            return []

# --- API Client ---
class MultiPlatformAPI:
    """API Client for GitHub and GitLab with caching and retry logic."""
    def __init__(self):
        self.client = None
        self.rate_limit_remaining = 5000



# [LLM NOTE] This is part 2 of 6 from the archive engine-archaeologist.py.
# [LLM NOTE] Lines 301 to 600 are included in this part.
# [LLM NOTE] Please retain awareness of the multi-part nature of this archive.

        """Save scan metadata and return scan ID."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO scans (total_forks, total_engines, duration)
                    VALUES (?, ?, ?)
                """, (scan_data["total_forks"], scan_data["total_engines"], scan_data["duration"])))
                scan_id = cursor.lastrowid
                await self.conn.commit()
                return scan_id
        except aiosqlite.Error as e:
            logger.error(f"Error saving scan: {e}")
            return -1
    
    async def save_fork(self, scan_id: int, fork: Dict) -> int:
        """Save fork analysis with state tracking."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT id, first_seen_scan_id, last_seen_scan_id
                    FROM forks
                    WHERE owner = ? AND repo = ?
                    ORDER BY scan_id DESC LIMIT 1
                """, (fork["owner"], fork["repo"])))
                existing_fork = await cursor.fetchone()
                
                if existing_fork:
                    fork_id, first_seen, last_seen = existing_fork
                    is_new = False
                    is_updated = await self.is_fork_updated(fork_id, fork)
                    is_removed = False
                    new_last_seen = scan_id
                else:
                    is_new = True
                    is_updated = False
                    is_removed = False
                    first_seen = scan_id
                    new_last_seen = scan_id
                
                if existing_fork:
                    await cursor.execute("""
                        UPDATE forks
                        SET scan_id = ?, last_activity = ?, total_score = ?, classification = ?,
                            stars = ?, forks_count = ?, open_issues = ?, fork_url = ?,
                            is_new = ?, is_updated = ?, is_removed = ?, last_seen_scan_id = ?
                        WHERE id = ?
                    """, (
                        scan_id, fork["last_activity"], fork["total_score"], fork["classification"],
                        fork["stars"], fork["forks"], fork["open_issues"], fork["fork_url"],
                        int(is_new), int(is_updated), int(is_removed), new_last_seen, fork_id
                    ))
                else:
                    await cursor.execute("""
                        INSERT INTO forks (
                            scan_id, owner, repo, platform, last_activity, total_score, classification,
                            stars, forks_count, open_issues, fork_url, is_new, is_updated, is_removed,
                            first_seen_scan_id, last_seen_scan_id
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        scan_id, fork["owner"], fork["repo"], fork["platform"], fork["last_activity"],
                        fork["total_score"], fork["classification"], fork["stars"], fork["forks"],
                        fork["open_issues"], fork["fork_url"], int(is_new), int(is_updated),
                        int(is_removed), first_seen, new_last_seen
                    ))
                    fork_id = cursor.lastrowid
                
                await self.conn.commit()
                return fork_id
        except aiosqlite.Error as e:
            logger.error(f"Error saving fork: {e}")
            return -1
    
    async def is_fork_updated(self, fork_id: int, new_fork: Dict) -> bool:
        """Check if a fork has significant updates."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT total_score, last_activity, classification
                    FROM forks
                    WHERE id = ?
                """, (fork_id,)))
                old_fork = await cursor.fetchone()
                
                if not old_fork:
                    return True
                
                old_score, old_activity, old_class = old_fork
                new_score = new_fork.get("total_score", 0)
                new_activity = new_fork.get("last_activity")
                new_class = new_fork.get("classification", "")
                
                if old_score > 0 and abs(new_score - old_score) / old_score > 0.1:
                    return True
                if new_class != old_class:
                    return True
                if new_activity and old_activity:
                    if (pendulum.parse(new_activity) - pendulum.parse(old_activity)).in_days() > settings.diff_days_threshold:
                        return True
                return False
        except aiosqlite.Error as e:
            logger.error(f"Error checking fork update: {e}")
            return True
    
    async def save_engine(self, fork_id: int, engine: Dict) -> int:
        """Save engine information with state tracking."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT id, first_seen_scan_id, last_seen_scan_id, file_hash
                    FROM engines
                    WHERE fork_id = ? AND name = ? AND path = ?
                """, (fork_id, engine["name"], engine["path"])))
                existing_engine = await cursor.fetchone()
                
                if existing_engine:
                    engine_id, first_seen, last_seen, old_file_hash = existing_engine
                    is_new = False
                    is_updated = await self.is_engine_updated(engine_id, engine)
                    is_removed = False
                    new_last_seen = last_seen
                else:
                    is_new = True
                    is_updated = False
                    is_removed = False
                    first_seen = None
                    new_last_seen = None
                
                file_hash = self.calculate_engine_hash(engine)
                
                if existing_engine:
                    await cursor.execute("""
                        UPDATE engines
                        SET size_bytes = ?, last_modified = ?, complexity_score = ?, classification = ?,
                            branch = ?, commit_hash = ?, author = ?, commit_message = ?, is_experimental = ?,
                            is_new = ?, is_updated = ?, is_removed = ?, validation_score = ?, file_hash = ?,
                            ast_signature = ?, similarity_score = ?, loc = ?, comment_lines = ?,
                            comment_ratio = ?, code_complexity = ?
                        WHERE id = ?
                    """, (
                        engine["size_bytes"], engine["last_modified"], engine["complexity_score"],
                        engine["classification"], engine["branch"], engine["commit_hash"], engine["author"],
                        engine["commit_message"], int(engine["is_experimental"]), int(is_new), int(is_updated),
                        int(is_removed), engine["validation_score"], file_hash, engine["ast_signature"],
                        engine["similarity_score"], engine["loc"], engine["comment_lines"],
                        engine["comment_ratio"], engine["code_complexity"], engine_id
                    ))
                else:
                    await cursor.execute("""
                        INSERT INTO engines (
                            fork_id, name, path, size_bytes, last_modified, complexity_score, classification,
                            branch, commit_hash, author, commit_message, is_experimental, is_new, is_updated,
                            is_removed, validation_score, file_hash, ast_signature, similarity_score,
                            loc, comment_lines, comment_ratio, code_complexity
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fork_id, engine["name"], engine["path"], engine["size_bytes"], engine["last_modified"],
                        engine["complexity_score"], engine["classification"], engine["branch"], engine["commit_hash"],
                        engine["author"], engine["commit_message"], int(engine["is_experimental"]), int(is_new),
                        int(is_updated), int(is_removed), engine["validation_score"], file_hash,
                        engine["ast_signature"], engine["similarity_score"], engine["loc"], engine["comment_lines"],
                        engine["comment_ratio"], engine["code_complexity"]
                    ))
                    engine_id = cursor.lastrowid
                
                for file_path in engine.get("files", []):
                    await self.save_engine_file(engine_id, file_path)
                
                await self.conn.commit()
                return engine_id
        except aiosqlite.Error as e:
            logger.error(f"Error saving engine: {e}")
            return -1
    
    async def is_engine_updated(self, engine_id: int, new_engine: Dict) -> bool:
        """Check if an engine has significant updates."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    SELECT complexity_score, validation_score, file_hash, last_modified
                    FROM engines
                    WHERE id = ?
                """, (engine_id,)))
                old_engine = await cursor.fetchone()
                
                if not old_engine:
                    return True
                
                old_score, old_validation, old_hash, old_modified = old_engine
                new_score = new_engine.get("complexity_score", 0)
                new_validation = new_engine.get("validation_score", 0)
                new_hash = self.calculate_engine_hash(new_engine)
                new_modified = new_engine.get("last_modified")
                
                if new_hash != old_hash:
                    return True
                if abs(new_score - old_score) > 10:
                    return True
                if abs(new_validation - old_validation) > 15:
                    return True
                if new_modified and old_modified:
                    if (pendulum.parse(new_modified) - pendulum.parse(old_modified)).in_days() > settings.diff_days_threshold:
                        return True
                return False
        except aiosqlite.Error as e:
            logger.error(f"Error checking engine update: {e}")
            return True
    
    def calculate_engine_hash(self, engine: Dict) -> str:
        """Calculate a hash based on engine file contents and paths."""
        file_data = "\n".join(sorted(f"{f}:{engine.get("content_" + f, "")}" for f in engine.get("files", [])))
        return hashlib.sha256(file_data.encode()).hexdigest()
    
    async def save_engine_file(self, engine_id: int, file_path: str):
        """Save engine file information."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT OR REPLACE INTO engine_files (engine_id, path)
                    VALUES (?, ?)
                """, (engine_id, file_path)))
                await self.conn.commit()
        except aiosqlite.Error as e:
            logger.error(f"Error saving engine file: {e}")
    
    async def save_differential(self, scan_id: int, diff_data: Dict):
        """Save differential analysis results."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO differential (
                        scan_id, new_forks, updated_forks, removed_forks,
                        new_engines, updated_engines, removed_engines
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    scan_id, diff_data["new_forks"], diff_data["updated_forks"], diff_data["removed_forks"],
                    diff_data["new_engines"], diff_data["updated_engines"], diff_data["removed_engines"]
                )))
                await self.conn.commit()
        except aiosqlite.Error as e:
            logger.error(f"Error saving differential data: {e}")
    
    async def mark_removed_entities(self, current_scan_id: int):
        """Mark forks and engines not seen in the current scan."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("""
                    UPDATE forks
                    SET is_removed = 1, last_seen_scan_id = ?
                    WHERE last_seen_scan_id < ? AND is_removed = 0
                """, (current_scan_id - 1, current_scan_id)))
                
                await cursor.execute("""
                    UPDATE engines
                    SET is_removed = 1, last_seen_scan_id = ?
                    WHERE last_seen_scan_id < ? AND is_removed = 0
                """, (current_scan_id - 1, current_scan_id)))
                
                await self.conn.commit()
        except aiosqlite.Error as e:
            logger.error(f"Error marking removed entities: {e}")
    
    async def get_last_scan(self):
        """Get the last scan from the database."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM scans ORDER BY timestamp DESC LIMIT 1")
                return await cursor.fetchone()
        except aiosqlite.Error as e:
            logger.error(f"Error getting last scan: {e}")
            return None
    
    async def get_forks_for_scan(self, scan_id: int):
        """Get all forks for a specific scan."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM forks WHERE scan_id = ?", (scan_id,)))
                return await cursor.fetchall()
        except aiosqlite.Error as e:
            logger.error(f"Error getting forks: {e}")
            return []
    
    async def get_engines_for_fork(self, fork_id: int):
        """Get all engines for a specific fork."""
        try:
            async with self.conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM engines WHERE fork_id = ?", (fork_id,)))
                return await cursor.fetchall()
        except aiosqlite.Error as e:
            logger.error(f"Error getting engines: {e}")
            return []

# --- API Client ---
class MultiPlatformAPI:
    """API Client for GitHub and GitLab with caching and retry logic."""
    def __init__(self):
        self.client = None
        self.rate_limit_remaining = 5000



# [LLM NOTE] This is part 3 of 6 from the archive engine-archaeologist.py.
# [LLM NOTE] Lines 601 to 900 are included in this part.
# [LLM NOTE] Please retain awareness of the multi-part nature of this archive.

        self.rate_limit_reset = 0
        self.cache = self._load_cache()
        self.etags = {}
        self.gitlab_project_id = None
    
    def _load_cache(self) -> Dict:
        """Load persistent cache from file."""
        try:
            if os.path.exists(settings.cache_file):
                with open(settings.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(settings.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            headers={"User-Agent": settings.user_agent},
            timeout=30,
            http2=True
        )
        if settings.gitlab_token:
            await self.fetch_gitlab_project_id()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._save_cache()
        await self.client.aclose()
    
    async def fetch_gitlab_project_id(self):
        """Fetch GitLab project ID for the main repository."""
        url = f"{settings.gitlab_api}/projects?search={settings.main_owner}%2F{settings.main_repo}"
        headers = {"Authorization": f"Bearer {settings.gitlab_token}"}
        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data:
                self.gitlab_project_id = data[0]["id"]
                logger.info(f"Found GitLab project ID: {self.gitlab_project_id}")
            else:
                logger.error("No GitLab project found for the main repository.")
        except httpx.HTTPError as e:
            logger.error(f"GitLab project ID fetch error: {e}")
    
    async def api_call(self, url: str, platform: str = "github", use_cache: bool = True, max_retries: int = 3) -> Optional[Dict]:
        """Make an API call with caching and retry logic."""
        cache_key = hashlib.sha256(url.encode()).hexdigest()
        
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        headers = {}
        if platform == "github":
            headers["Authorization"] = f"token {settings.github_token}"
        elif platform == "gitlab" and settings.gitlab_token:
            headers["Authorization"] = f"Bearer {settings.gitlab_token}"
        
        if cache_key in self.etags:
            headers["If-None-Match"] = self.etags[cache_key]
        
        for attempt in range(max_retries):
            try:
                response = await self.client.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    self.cache[cache_key] = data
                    if "ETag" in response.headers:
                        self.etags[cache_key] = response.headers["ETag"]
                    return data
                elif response.status_code == 304:
                    return self.cache.get(cache_key)
                elif response.status_code == 403:
                    self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    sleep_time = max(0, reset_time - time.time()) + random.uniform(1, 3)
                    logger.warning(f"Rate limit hit. Sleeping for {sleep_time:.1f} seconds.")
                    await asyncio.sleep(sleep_time)
                    continue
                elif response.status_code == 404:
                    logger.debug(f"Not found: {url}")
                    return None
                elif response.status_code >= 500:
                    sleep_time = (2 ** attempt) + random.random()
                    logger.warning(f"Server error {response.status_code}: {url}. Retrying in {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    logger.warning(f"API Error {response.status_code}: {url}")
                    return None
            except httpx.HTTPError as e:
                logger.error(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) + random.random()
                    await asyncio.sleep(sleep_time)
                    continue
                return None
        return None
    
    async def get_forks(self, platform: str = "github", days: int = 365) -> List[Dict]:
        """Get forks with recent activity."""
        forks = []
        cutoff_date = pendulum.now().subtract(days=days)
        
        if platform == "github":
            page = 1
            per_page = 100
            while True:
                url = f"{settings.github_api}/repos/{settings.main_owner}/{settings.main_repo}/forks?per_page={per_page}&page={page}&sort=updated"
                data = await self.api_call(url, platform)
                if not data:
                    break
                for fork in data:
                    if fork.get("pushed_at"):
                        pushed_date = pendulum.parse(fork["pushed_at"])
                        if pushed_date > cutoff_date:
                            forks.append({
                                "id": fork["id"],
                                "name": fork["name"],
                                "owner": fork["owner"]["login"],
                                "pushed_at": fork["pushed_at"],
                                "html_url": fork["html_url"],
                                "stargazers_count": fork.get("stargazers_count", 0),
                                "forks_count": fork.get("forks_count", 0),
                                "open_issues_count": fork.get("open_issues_count", 0)
                            })
                if len(data) < per_page:
                    break
                page += 1
                await asyncio.sleep(0.5)
            logger.info(f"Found {len(forks)} active forks on GitHub")
            return forks
        
        elif platform == "gitlab" and settings.gitlab_token:
            page = 1
            per_page = 100
            while True:
                url = f"{settings.gitlab_api}/projects/{self.gitlab_project_id}/forks?per_page={per_page}&page={page}"
                data = await self.api_call(url, "gitlab")
                if not data:
                    break
                for fork in data:
                    if fork.get("last_activity_at"):
                        last_activity = pendulum.parse(fork["last_activity_at"])
                        if last_activity > cutoff_date:
                            forks.append({
                                "id": fork["id"],
                                "name": fork["name"],
                                "owner": fork["namespace"]["path"],
                                "pushed_at": fork["last_activity_at"],
                                "html_url": fork["web_url"],
                                "stargazers_count": fork.get("star_count", 0),
                                "forks_count": fork.get("forks_count", 0),
                                "open_issues_count": fork.get("open_issues_count", 0)
                            })
                if len(data) < per_page:
                    break
                page += 1
                await asyncio.sleep(0.5)
            logger.info(f"Found {len(forks)} active forks on GitLab")
            return forks
        return []
    
    async def get_branches(self, owner: str, repo: str, platform: str = "github") -> List[str]:
        """Get branches for a repository."""
        if platform == "github":
            url = f"{settings.github_api}/repos/{owner}/{repo}/branches"
            data = await self.api_call(url, platform)
            return [b["name"] for b in data] if data else []
        
        elif platform == "gitlab" and settings.gitlab_token:
            project_id = await self.get_gitlab_project_id(owner, repo)
            if not project_id:
                return []
            url = f"{settings.gitlab_api}/projects/{project_id}/repository/branches"
            data = await self.api_call(url, "gitlab")
            return [b["name"] for b in data] if data else []
        return []
    
    async def get_gitlab_project_id(self, owner: str, repo: str) -> Optional[int]:
        """Get GitLab project ID for a fork."""
        url = f"{settings.gitlab_api}/projects?search={owner}%2F{repo}"
        data = await self.api_call(url, "gitlab")
        if data and len(data) > 0 and "id" in data[0]:
            return data[0]["id"]
        return None
    
    async def get_repo_tree(self, owner: str, repo: str, branch: str, platform: str = "github") -> Optional[Dict]:
        """Get full repository tree."""
        if platform == "github":
            url = f"{settings.github_api}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
            return await self.api_call(url, platform)
        elif platform == "gitlab" and settings.gitlab_token:
            project_id = await self.get_gitlab_project_id(owner, repo)
            if not project_id:
                return None
            url = f"{settings.gitlab_api}/projects/{project_id}/repository/tree?recursive=true&ref={branch}"
            return await self.api_call(url, "gitlab")
        return None
    
    async def get_partial_repo_tree(self, owner: str, repo: str, branch: str, platform: str = "github") -> Optional[Dict]:
        """Get partial repository tree for large repositories with caching."""
        cache_key = f"partial_tree_{owner}_{repo}_{branch}_{platform}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if platform == "github":
            url = f"{settings.github_api}/repos/{owner}/{repo}/git/trees/{branch}"
            root_tree = await self.api_call(url, platform)
            if not root_tree or "tree" not in root_tree:
                return None
            
            engine_dirs = [item["path"] for item in root_tree["tree"] if item["type"] == "tree" and ("engine" in item["path"].lower() or "src" in item["path"].lower())]
            all_tree = []
            for path in engine_dirs:
                url = f"{settings.github_api}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1&path={path}"
                tree_data = await self.api_call(url, platform)
                if tree_data and "tree" in tree_data:
                    all_tree.extend(tree_data["tree"])
            
            result = {"tree": all_tree}
            self.cache[cache_key] = result
            return result
        
        elif platform == "gitlab" and settings.gitlab_token:
            project_id = await self.get_gitlab_project_id(owner, repo)
            if not project_id:
                return None
            url = f"{settings.gitlab_api}/projects/{project_id}/repository/tree?ref={branch}"
            root_tree = await self.api_call(url, "gitlab")
            if not root_tree:
                return None
            
            engine_dirs = [item["path"] for item in root_tree if item["type"] == "tree" and ("engine" in item["path"].lower() or "src" in item["path"].lower())]
            all_tree = []
            for path in engine_dirs:
                url = f"{settings.gitlab_api}/projects/{project_id}/repository/tree?recursive=true&ref={branch}&path={path}"
                tree_data = await self.api_call(url, "gitlab")
                if tree_data:
                    all_tree.extend(tree_data)
            
            result = {"tree": all_tree}
            self.cache[cache_key] = result
            return result
        return None
    
    async def get_commit_history(self, owner: str, repo: str, path: str, platform: str = "github") -> Optional[Dict]:
        """Get latest commit history for a path."""
        if platform == "github":
            url = f"{settings.github_api}/repos/{owner}/{repo}/commits?path={path}&per_page=1"
            data = await self.api_call(url, platform)
            return data[0] if data and len(data) > 0 else None
        
        elif platform == "gitlab" and settings.gitlab_token:
            project_id = await self.get_gitlab_project_id(owner, repo)
            if not project_id:
                return None
            url = f"{settings.gitlab_api}/projects/{project_id}/repository/commits?path={path}&per_page=1"
            data = await self.api_call(url, "gitlab")
            return data[0] if data and len(data) > 0 else None
        return None
    
    async def get_file_content(self, owner: str, repo: str, path: str, platform: str = "github") -> Optional[str]:
        """Get file content."""
        if platform == "github":
            url = f"{settings.github_api}/repos/{owner}/{repo}/contents/{path}"
            data = await self.api_call(url, platform)
            if data and "content" in data:
                return base64.b64decode(data["content"]).decode('utf-8', errors='ignore')
            return None
        
        elif platform == "gitlab" and settings.gitlab_token:
            project_id = await self.get_gitlab_project_id(owner, repo)
            if not project_id:
                return None
            url = f"{settings.gitlab_api}/projects/{project_id}/repository/files/{path}?ref=master"
            data = await self.api_call(url, "gitlab")
            if data and "content" in data:
                return base64.b64decode(data["content"]).decode('utf-8', errors='ignore')
        return None

# --- Engine Analyzer ---
class EngineAnalyzer:
    """Class for detecting and classifying engines using AST and ML."""
    def __init__(self, api: MultiPlatformAPI):
        self.api = api
        self.known_engines: Set[str] = set()
        self.engine_patterns = [
            r"engines/([^/]+)/.*\.(cpp|h|mk)$",
            r"([^/]+)engine\.(cpp|h)$",
            r"([^/]+)/engine\.(cpp|h)$",



# [LLM NOTE] This is part 4 of 6 from the archive engine-archaeologist.py.
# [LLM NOTE] Lines 901 to 1200 are included in this part.
# [LLM NOTE] Please retain awareness of the multi-part nature of this archive.

            r"src/([^/]+)/.*engine.*\.(cpp|h)$"
        ]
        self.key_files = ["detection.cpp", "detection.h", "engine.cpp", "engine.h", "module.mk"]
        self.engine_keywords = ["engine", "detection", "scummvm", "game", "adapter"]
        self.false_positive_dirs = ["test", "doc", "example", "samples", "tools", "util"]
        
        self.ast_parser, self.ast_queries = initialize_ast_parser()
        self.engine_signatures = self.load_known_engine_signatures()
        self.ml_model = None
        self.ml_encoder = None
        self.load_ml_model()
    
    async def load_known_engines(self):
        """Load known engines from the main repository."""
        tree_data = await self.api.get_repo_tree(settings.main_owner, settings.main_repo, "master")
        if not tree_data or "tree" not in tree_data:
            return
        
        for item in tree_data["tree"]:
            for pattern in self.engine_patterns:
                match = re.search(pattern, item["path"])
                if match:
                    engine_name = match.group(1).lower()
                    if not any(fp in engine_name for fp in self.false_positive_dirs):
                        self.known_engines.add(engine_name)
        logger.info(f"Loaded {len(self.known_engines)} known engines from main repository")
    
    def load_known_engine_signatures(self) -> Dict[str, List[str]]:
        """Load AST signatures of known engines."""
        return {
            "scumm": ["ScummEngine::ScummEngine", "REGISTER_ENGINE(Scumm)"],
            "agi": ["AgiEngine::AgiEngine", "REGISTER_ENGINE(Agi)"],
            "sky": ["SkyEngine::SkyEngine", "REGISTER_ENGINE(Sky)"],
            "queen": ["QueenEngine::QueenEngine", "REGISTER_ENGINE(Queen)"],
            "saga": ["SagaEngine::SagaEngine", "REGISTER_ENGINE(Saga)"]
        }
    
    def load_ml_model(self):
        """Load trained ML model if available."""
        if not ML_AVAILABLE or not Path(settings.ml_model_path).exists():
            return
        try:
            with open(settings.ml_model_path, "rb") as f:
                self.ml_model, self.ml_encoder = pickle.load(f)
            logger.info("Loaded trained ML model")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
    
    async def analyze_ast(self, content: str) -> List[Dict]:
        """Analyze source code using Tree-sitter AST."""
        if not self.ast_parser or not content:
            return []
        try:
            tree = self.ast_parser.parse(bytes(content, "utf8"))
            results = []
            
            for query_name, query in self.ast_queries.items():
                captures = query.captures(tree.root_node)
                for node, name in captures:
                    if query_name == "engine_inheritance" and name == "namespace" and node.text.decode() == "ScummVM":
                        class_node = next((n for n, n_name in captures if n_name == "class_name" and n.parent == node.parent), None)
                        base_node = next((n for n, n_name in captures if n_name == "base_class" and n.parent == node.parent), None)
                        if class_node and base_node and base_node.text.decode() == "Engine":
                            results.append({"type": "engine_inheritance", "class": class_node.text.decode(), "base": base_node.text.decode()})
                    elif query_name == "engine_registration" and name == "func_name" and node.text.decode() == "REGISTER_ENGINE":
                        engine_name_node = next((n for n, n_name in captures if n_name == "engine_name" and n.parent == node.parent), None)
                        if engine_name_node:
                            results.append({"type": "engine_registration", "engine": engine_name_node.text.decode().strip("\"")})
                    elif query_name == "plugin_registration" and name == "func_name" and node.text.decode() == "REGISTER_PLUGIN":
                        plugin_name_node = next((n for n, n_name in captures if n_name == "plugin_name" and n.parent == node.parent), None)
                        if plugin_name_node:
                            results.append({"type": "plugin_registration", "plugin": plugin_name_node.text.decode().strip("\"")})
                    elif query_name == "engine_detection" and name == "struct_name" and "detect" in node.text.decode().lower():
                        results.append({"type": "engine_detection", "struct": node.text.decode()})
            return results
        except Exception as e:
            logger.error(f"AST analysis error: {e}")
            return []
    
    async def calculate_similarity(self, engine_name: str, ast_results: List[Dict]) -> float:
        """Calculate similarity to known engines."""
        if engine_name in self.engine_signatures:
            expected = set(self.engine_signatures[engine_name])
            found = {f"{r["class"]}" if r["type"] == "engine_inheritance" else f"REGISTER_ENGINE({r["engine"]})" for r in ast_results}
            intersection = len(expected & found)
            union = len(expected | found)
            return intersection / union if union > 0 else 0.0
        return 0.0
    
    def calculate_code_metrics(self, content: str) -> Dict[str, float]:
        """Calculate code quality metrics."""
        if not content:
            return {"loc": 0, "comment_lines": 0, "comment_ratio": 0.0, "complexity": 0}
        
        lines = content.splitlines()
        loc = 0
        comment_lines = 0
        in_block_comment = False
        complexity = 0
        control_flow_keywords = ["if", "else", "for", "while", "switch", "case"]
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if in_block_comment:
                comment_lines += 1
                if "*/" in stripped:
                    in_block_comment = False
                continue
            if stripped.startswith("/*"):
                comment_lines += 1
                in_block_comment = True
                if "*/" in stripped:
                    in_block_comment = False
                continue
            if stripped.startswith("//"):
                comment_lines += 1
                continue
            loc += 1
            if any(keyword in stripped for keyword in control_flow_keywords):
                complexity += 1
            if "//" in stripped:
                comment_lines += 1
        
        total_lines = loc + comment_lines
        comment_ratio = (comment_lines / total_lines) * 100 if total_lines > 0 else 0.0
        return {"loc": loc, "comment_lines": comment_lines, "comment_ratio": comment_ratio, "complexity": complexity}
    
    def aggregate_code_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate code metrics from multiple files."""
        if not metrics_list:
            return {}
        return {
            "total_loc": sum(m.get("loc", 0) for m in metrics_list),
            "total_comment_lines": sum(m.get("comment_lines", 0) for m in metrics_list),
            "avg_comment_ratio": np.mean([m.get("comment_ratio", 0) for m in metrics_list]) if metrics_list else 0,
            "total_complexity": sum(m.get("complexity", 0) for m in metrics_list)
        }
    
    async def ml_classify_engine(self, engine_info: Dict) -> str:
        """Classify engine using ML or fallback to rule-based."""
        if not self.ml_model or not ML_AVAILABLE:
            return self.classify_engine(engine_info["complexity_score"], len(engine_info.get("files", [])))
        
        try:
            features = np.array([
                engine_info["complexity_score"],
                len(engine_info.get("files", [])),
                engine_info.get("validation_score", 0),
                engine_info.get("similarity_score", 0),
                engine_info.get("code_metrics", {}).get("total_loc", 0),
                engine_info.get("code_metrics", {}).get("avg_comment_ratio", 0),
                engine_info.get("code_metrics", {}).get("total_complexity", 0)
            ]).reshape(1, -1)
            prediction = self.ml_model.predict(features)
            return self.ml_encoder.inverse_transform(prediction)[0]
        except Exception as e:
            logger.error(f"ML classification failed: {e}")
            return self.classify_engine(engine_info["complexity_score"], len(engine_info.get("files", [])))
    
    async def analyze_repository(self, owner: str, repo: str, branch: str, platform: str = "github") -> List[Dict]:
        """Analyze repository for engines."""
        engines = []
        tree_data = await self.api.get_partial_repo_tree(owner, repo, branch, platform) or await self.api.get_repo_tree(owner, repo, branch, platform)
        if not tree_data or "tree" not in tree_data:
            return engines
        
        engine_dirs = {}
        for item in tree_data["tree"]:
            if item.get("type") != "blob":
                continue
            path = item["path"]
            for pattern in self.engine_patterns:
                match = re.search(pattern, path)
                if match:
                    engine_name = match.group(1).lower()
                    if any(fp in engine_name for fp in self.false_positive_dirs):
                        continue
                    if engine_name not in engine_dirs:
                        engine_dirs[engine_name] = []
                    engine_dirs[engine_name].append({
                        "path": path,
                        "size": item.get("size", 0),
                        "sha": item.get("sha", "")
                    })
        
        for engine_name, files in engine_dirs.items():
            if len(files) < 2:
                continue
            
            total_size = sum(f["size"] for f in files)
            file_paths = [f["path"] for f in files]
            last_commit = await self.api.get_commit_history(owner, repo, f"engines/{engine_name}" if "engines/" in file_paths[0] else engine_name, platform)
            last_modified = last_commit["commit"]["committer"]["date"] if last_commit else pendulum.now().isoformat()
            
            validation_score = 0
            ast_results = []
            similarity_score = 0.0
            code_metrics = []
            contents = {}
            
            for file in files:
                if any(key_file in file["path"] for key_file in self.key_files):
                    content = await self.api.get_file_content(owner, repo, file["path"], platform)
                    if content:
                        contents[file["path"]] = content
                        file_ast = await self.analyze_ast(content)
                        ast_results.extend(file_ast)
                        validation_score += 10
                        code_metrics.append(self.calculate_code_metrics(content))
            
            if ast_results:
                similarity_score = await self.calculate_similarity(engine_name, ast_results)
                validation_score += int(similarity_score * 30)
            
            if validation_score < 20:
                logger.debug(f"Skipping engine {engine_name} (validation score: {validation_score})")
                continue
            
            aggregated_metrics = self.aggregate_code_metrics(code_metrics)
            engine_info = {
                "files": file_paths,
                "size_bytes": total_size,
                "last_modified": last_modified,
                "validation_score": validation_score,
                "code_metrics": aggregated_metrics
            }
            complexity = self.calculate_complexity_score(engine_info)
            classification = await self.ml_classify_engine({**engine_info, "complexity_score": complexity, "similarity_score": similarity_score})
            
            engines.append({
                "name": engine_name,
                "path": f"engines/{engine_name}" if "engines/" in file_paths[0] else engine_name,
                "files": file_paths,
                "size_bytes": total_size,
                "last_modified": last_modified,
                "complexity_score": complexity,
                "classification": classification,
                "branch": branch,
                "commit_hash": last_commit.get("sha", "") if last_commit else "",
                "author": last_commit["commit"]["author"]["name"] if last_commit else "",
                "commit_message": last_commit["commit"]["message"] if last_commit else "",
                "is_experimental": "experimental" in engine_name.lower(),
                "validation_score": validation_score,
                "ast_signature": json.dumps(ast_results),
                "similarity_score": similarity_score,
                "loc": aggregated_metrics.get("total_loc", 0),
                "comment_lines": aggregated_metrics.get("total_comment_lines", 0),
                "comment_ratio": aggregated_metrics.get("avg_comment_ratio", 0.0),
                "code_complexity": aggregated_metrics.get("total_complexity", 0),
                **{f"content_{path}": contents.get(path, "") for path in file_paths}
            })
        return engines
    
    def calculate_complexity_score(self, engine_info: Dict) -> int:
        """Calculate complexity score."""
        score = min(len(engine_info.get("files", [])) * 2, 50)
        score += min(engine_info.get("size_bytes", 0) // 1000, 30)
        score += len({Path(f).suffix.lower() for f in engine_info.get("files", [])}) * 5
        for key_file in self.key_files:
            if any(key_file in f for f in engine_info.get("files", [])):
                score += 10
        last_modified = pendulum.parse(engine_info.get("last_modified", pendulum.now().isoformat()))
        age_days = (pendulum.now() - last_modified).in_days()
        if age_days < 180:
            score += min(20, (180 - age_days) // 10)
        score += min(engine_info.get("validation_score", 0), 30)
        comment_ratio = engine_info.get("code_metrics", {}).get("avg_comment_ratio", 0)
        if comment_ratio > 20:
            score += 15
        elif comment_ratio > 10:
            score += 10
        elif comment_ratio > 5:
            score += 5
        complexity = engine_info.get("code_metrics", {}).get("total_complexity", 0)
        if complexity > 50:
            score += 10
        elif complexity > 20:
            score += 5
        return min(score, 100)
    
    def classify_engine(self, score: int, file_count: int) -> str:
        """Rule-based engine classification."""
        if score >= 90 and file_count >= 20:
            return "💎 DIAMOND"
        elif score >= 70 and file_count >= 15:
            return "🥇 GOLD"
        elif score >= 50 and file_count >= 10:
            return "🥈 SILVER"
        elif score >= 30 and file_count >= 5:
            return "🥉 BRONZE"
        elif score >= 15 and file_count >= 2:
            return "⚡ SPARK"
        else:
            return "🔍 RESEARCH"
    
    async def train_classifier(self):
        """Train the ML classifier with data from the database."""
        if not ML_AVAILABLE:



# [LLM NOTE] This is part 5 of 6 from the archive engine-archaeologist.py.
# [LLM NOTE] Lines 1201 to 1500 are included in this part.
# [LLM NOTE] Please retain awareness of the multi-part nature of this archive.

            logger.error("Scikit-learn not available for ML training.")
            return
        try:
            async with aiosqlite.connect(settings.db_file) as conn:
                cursor = await conn.cursor()
                await cursor.execute("""
                    SELECT complexity_score, size_bytes, validation_score, similarity_score, loc, comment_ratio, code_complexity, classification
                    FROM engines
                    WHERE is_removed = 0 AND complexity_score IS NOT NULL
                """
                data = await cursor.fetchall()
            
            if not data:
                logger.error("No data available for ML training.")
                return
            
            X = np.array([[row[0], row[1], row[2], row[3], row[4], row[5], row[6]] for row in data])
            y = [row[7] for row in data]
            
            self.ml_encoder = LabelEncoder()
            y_encoded = self.ml_encoder.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ml_model.fit(X_train, y_train)
            
            accuracy = self.ml_model.score(X_test, y_test)
            logger.info(f"ML model trained with accuracy: {accuracy:.2f}")
            
            with open(settings.ml_model_path, "wb") as f:
                pickle.dump((self.ml_model, self.ml_encoder), f)
        except Exception as e:
            logger.error(f"Failed to train ML model: {e}")

# --- Fork Scanner ---
class AdvancedForkScanner:
    """Class for scanning forks with concurrency control."""
    def __init__(self, api: MultiPlatformAPI, db: Database):
        self.api = api
        self.analyzer = EngineAnalyzer(api)
        self.db = db
        self.results: List[ForkAnalysis] = []
        self.scan_stats = {
            "forks_scanned": 0, "engines_found": 0, "new_forks": 0, "updated_forks": 0,
            "removed_forks": 0, "new_engines": 0, "updated_engines": 0, "removed_engines": 0,
            "start_time": pendulum.now(), "end_time": None, "duration": 0.0,
            "classification_counts": defaultdict(int)
        }
    
    async def scan_forks(self, days: int = 365, all_branches: bool = False, max_forks: int = 100, quiet: bool = False):
        """Scan forks with semaphore for concurrency."""
        await self.analyzer.load_known_engines()
        github_forks = await self.api.get_forks("github", days)
        gitlab_forks = await self.api.get_forks("gitlab", days) if settings.gitlab_token else []
        forks = github_forks + gitlab_forks
        total_forks = min(len(forks), max_forks)
        
        if not forks:
            logger.info("No active forks found")
            return []
        
        semaphore = asyncio.Semaphore(settings.max_concurrent_scans)
        tasks = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=quiet
        ) as progress:
            task = progress.add_task("[cyan]Scanning forks...", total=total_forks)
            
            for i, fork in enumerate(forks[:max_forks]):
                tasks.append(asyncio.create_task(self.scan_fork(fork, "github" if "owner" in fork else "gitlab", all_branches, semaphore, i + 1, total_forks, progress, task)))
            
            self.results = [r for r in await asyncio.gather(*tasks) if r]
        
        self.scan_stats.update({
            "end_time": pendulum.now(),
            "forks_scanned": len(self.results),
            "engines_found": sum(len(fork.engines) for fork in self.results),
            "duration": (pendulum.now() - self.scan_stats["start_time"]).in_seconds()
        })
        
        for fork in self.results:
            for engine in fork.engines:
                self.scan_stats["classification_counts"][engine.classification] += 1
        
        scan_id = await self.db.save_scan({
            "total_forks": self.scan_stats["forks_scanned"],
            "total_engines": self.scan_stats["engines_found"],
            "duration": self.scan_stats["duration"]
        })
        
        for fork in self.results:
            fork_data = fork.dict()
            fork_id = await self.db.save_fork(scan_id, fork_data)
            for engine in fork.engines:
                await self.db.save_engine(fork_id, engine.dict())
        
        diff_data = await self.differential_analysis(scan_id)
        await self.db.save_differential(scan_id, diff_data)
        self.scan_stats.update(diff_data)
        await self.db.mark_removed_entities(scan_id)
        
        if not quiet:
            display_results(self.results, self.scan_stats)
        return self.results
    
    async def scan_fork(self, fork: Dict, platform: str, all_branches: bool, semaphore: asyncio.Semaphore, current: int, total: int, progress: Progress, task: int) -> Optional["ForkAnalysis"]:
        """Scan a single fork with timeout and retry."""
        async with semaphore:
            try:
                async with asyncio.timeout(settings.scan_timeout):
                    owner = fork["owner"]
                    repo = fork["name"]
                    fork_url = fork["html_url"]
                    stars = fork["stargazers_count"]
                    forks_count = fork["forks_count"]
                    open_issues = fork["open_issues_count"]
                    last_activity = fork["pushed_at"]
                    
                    progress.update(task, description=f"[cyan]Scanning {owner}/{repo} ({current}/{total})")
                    
                    branches = ["main", "master"]
                    if all_branches:
                        all_branches_list = await self.api.get_branches(owner, repo, platform)
                        branches.extend(self.prioritize_branches(all_branches_list))
                    branches = list(set(branches))[:settings.max_branches_per_fork]
                    
                    all_engines = []
                    analyzed_branches = []
                    notes = []
                    
                    for branch in branches:
                        for attempt in range(2):
                            try:
                                engines = await self.analyzer.analyze_repository(owner, repo, branch, platform)
                                if engines:
                                    all_engines.extend(engines)
                                    analyzed_branches.append(branch)
                                    notes.append(f"Found {len(engines)} engines in branch \'{branch}\'")
                                break
                            except Exception as e:
                                logger.debug(f"Error analyzing branch {branch} (attempt {attempt + 1}): {e}")
                                if attempt == 1:
                                    notes.append(f"Failed to analyze branch \'{branch}\'")
                                await asyncio.sleep(1)
                    
                    new_engines = [EngineInfo(**e) for e in all_engines if e["name"] not in self.analyzer.known_engines]
                    total_score = sum(e.complexity_score for e in new_engines)
                    classification = "❌ NO NEW ENGINES" if not new_engines else (
                        "🏆 HIGH VALUE" if total_score / len(new_engines) >= 70 else
                        "⭐ MEDIUM VALUE" if total_score / len(new_engines) >= 50 else
                        "📋 LOW VALUE" if total_score / len(new_engines) >= 30 else "🔍 EXPERIMENTAL"
                    ) if new_engines else "❌ NO NEW ENGINES"
                    
                    return ForkAnalysis(
                        owner=owner, repo=repo, platform=platform, last_activity=last_activity,
                        branches_analyzed=analyzed_branches, engines=new_engines, total_score=total_score,
                        classification=classification, notes=notes, fork_url=fork_url, stars=stars,
                        forks=forks_count, open_issues=open_issues, last_commit_date=pendulum.parse(last_activity)
                    )
            except asyncio.TimeoutError:
                logger.error(f"Scan of {fork["owner"]}/{fork["name"]} timed out")
                return None
            except Exception as e:
                logger.error(f"Error scanning fork {fork["owner"]}/{fork["name"]}: {e}")
                return None
    
    def prioritize_branches(self, branches: List[str]) -> List[str]:
        """Prioritize branches based on naming patterns."""
        priority_order = [r"main", r"master", r"develop", r"dev", r"feature/.+", r"engine/.+", r"experimental/.+"]
        prioritized = []
        for pattern in priority_order:
            regex = re.compile(pattern, re.IGNORECASE)
            prioritized.extend(b for b in branches if regex.match(b) and b not in prioritized)
        prioritized.extend(b for b in branches if b not in prioritized)
        return prioritized
    
    async def differential_analysis(self, current_scan_id: int) -> Dict:
        """Compare current scan with previous scan."""
        prev_scan = await self.db.get_last_scan()
        if not prev_scan or prev_scan[0] == current_scan_id:
            return {k: 0 for k in ["new_forks", "updated_forks", "removed_forks", "new_engines", "updated_engines", "removed_engines"]}
        
        prev_scan_id = prev_scan[0]
        prev_forks = await self.db.get_forks_for_scan(prev_scan_id)
        current_forks = await self.db.get_forks_for_scan(current_scan_id)
        
        prev_forks_dict = {(f[2], f[3]): f for f in prev_forks}
        current_forks_dict = {(f[2], f[3]): f for f in current_forks}
        
        new_forks = len([f for key, f in current_forks_dict.items() if key not in prev_forks_dict])
        updated_forks = len([f for key, f in current_forks_dict.items() if key in prev_forks_dict and f[7]])
        removed_forks = len([f for key, f in prev_forks_dict.items() if key not in current_forks_dict])
        
        new_engines = updated_engines = removed_engines = 0
        for current_fork in current_forks:
            prev_fork = next((f for f in prev_forks if f[2] == current_fork[2] and f[3] == current_fork[3]), None)
            if not prev_fork:
                continue
            current_engines = await self.db.get_engines_for_fork(current_fork[0])
            prev_engines = await self.db.get_engines_for_fork(prev_fork[0])
            
            prev_engines_dict = {e[2]: e for e in prev_engines}
            current_engines_dict = {e[2]: e for e in current_engines}
            
            new_engines += len([e for e in current_engines_dict if e not in prev_engines_dict])
            updated_engines += len([e for e in current_engines_dict.values() if e[2] in prev_engines_dict and e[13]])
            removed_engines += len([e for e in prev_engines_dict if e not in current_engines_dict])
        
        return {
            "new_forks": new_forks, "updated_forks": updated_forks, "removed_forks": removed_forks,
            "new_engines": new_engines, "updated_engines": updated_engines, "removed_engines": removed_engines
        }

# --- Data Models ---
class EngineInfo(BaseModel):
    name: str
    path: str
    files: List[str] = []
    size_bytes: int = 0
    last_modified: str = Field(default_factory=lambda: pendulum.now().isoformat())
    complexity_score: conint(ge=0, le=100) = 0
    classification: str = "🔍 RESEARCH"
    branch: str = "main"
    commit_hash: str = ""
    author: str = ""
    commit_message: str = ""
    is_experimental: bool = False
    validation_score: int = 0
    ast_signature: str = ""
    similarity_score: confloat(ge=0.0, le=1.0) = 0.0
    loc: int = 0
    comment_lines: int = 0
    comment_ratio: confloat(ge=0.0, le=100.0) = 0.0
    code_complexity: int = 0

class ForkAnalysis(BaseModel):
    owner: str
    repo: str
    platform: str
    last_activity: str
    branches_analyzed: List[str] = []
    engines: List[EngineInfo] = []
    total_score: int = 0
    classification: str = "❌ NO NEW ENGINES"
    notes: List[str] = []
    fork_url: str = ""
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    last_commit_date: datetime

# --- Web Service ---
if WEB_SERVICE_AVAILABLE:
    app = FastAPI(title="ScummVM Engine Discovery", version="6.0")
    sio = AsyncServer(async_mode="asgi", cors_allowed_origins="*")
    socketio_app = ASGIApp(sio)
    app.mount("/socket.io", socketio_app)
    
    scanner: Optional[AdvancedForkScanner] = None
    db = Database()
    scan_in_progress = False
    
    class ScanParams(BaseModel):
        days: conint(ge=1) = 365
        all_branches: bool = False
        max_forks: conint(ge=1) = 100
    
    async def progress_callback(message: str, current: int, total: int):
        """Send progress updates via WebSocket."""
        progress_percent = int((current / total) * 100) if total > 0 else 0
        await sio.emit("progress", {"message": message, "current": current, "total": total, "percent": progress_percent}))
    
    @app.get("/")
    async def dashboard():
        """Render the main dashboard."""
        return {
            "html": """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>ScummVM Engine Discovery</title>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
                    .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
                    .card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
                    .progress-bar { height: 20px; background: #ddd; border-radius: 5px; }
                    .progress-fill { height: 100%; background: #4caf50; width: 0%; border-radius: 5px; }
                    .stats { display: flex; flex-wrap: wrap; gap: 20px; }
                    .stat-card { flex: 1; min-width: 150px; padding: 10px; background: #f9f9f9; border-radius: 5px; }
                    .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                    .btn:hover { background: #2980b9; }
                    .chart-container { height: 300px; margin-top: 20px; }
                </style>



# [LLM NOTE] This is part 6 of 6 from the archive engine-archaeologist.py.
# [LLM NOTE] Lines 1501 to 1680 are included in this part.
# [LLM NOTE] Please retain awareness of the multi-part nature of this archive.

            </head>
            <body>
                <div class="header">
                    <h1>🎮 ScummVM Engine Discovery Dashboard</h1>
                </div>
                <div class="card">
                    <h2>Scan Control</h2>
                    <button class="btn" id="startScan">Start Scan</button>
                    <div class="progress-bar" style="margin-top: 10px;">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <p id="progressText">Ready to start scan</p>
                </div>
                <div class="card">
                    <h2>Scan Statistics</h2>
                    <div class="stats">
                        <div class="stat-card"><h3>Forks Scanned</h3><p id="forksScanned">0</p></div>
                        <div class="stat-card"><h3>Engines Found</h3><p id="enginesFound">0</p></div>
                        <div class="stat-card"><h3>New Engines</h3><p id="newEngines">0</p></div>
                    </div>
                </div>
                <div class="card">
                    <h2>Classification Distribution</h2>
                    <div class="chart-container"><canvas id="classificationChart"></canvas></div>
                </div>
                <script>
                    const socket = io();
                    let classificationChart = null;
                    
                    function initChart() {
                        const ctx = document.getElementById("classificationChart").getContext("2d");
                        classificationChart = new Chart(ctx, {
                            type: "pie",
                            data: { labels: [], datasets: [{ data: [], backgroundColor: ["#ff6384", "#36a2eb", "#ffce56", "#4bc0c0", "#9966ff"] }] },
                            options: { responsive: true, maintainAspectRatio: false }
                        });
                    }
                    
                    socket.on("progress", (data) => {
                        document.getElementById("progressFill").style.width = data.percent + "%";
                        document.getElementById("progressText").textContent = `${data.message} (${data.current}/${data.total}) - ${data.percent}%`;
                    });
                    
                    socket.on("stats", (data) => {
                        document.getElementById("forksScanned").textContent = data.forks_scanned;
                        document.getElementById("enginesFound").textContent = data.engines_found;
                        document.getElementById("newEngines").textContent = data.new_engines;
                        if (classificationChart) {
                            classificationChart.data.labels = Object.keys(data.classification_counts);
                            classificationChart.data.datasets[0].data = Object.values(data.classification_counts);
                            classificationChart.update();
                        }
                    });
                    
                    socket.on("scan_complete", (data) => {
                        console.log("Scan completed:", data);
                    });
                    
                    socket.on("error", (data) => {
                        alert(data.message);
                    });
                    
                    document.getElementById("startScan").addEventListener("click", () => {
                        const params = {
                            days: 365,
                            all_branches: false,
                            max_forks: 100
                        };
                        socket.emit("start_scan", params);
                    });
                    
                    initChart();
                </script>
            </body>
            </html>
            """
        }
    
    @sio.event
    async def start_scan(sid, data):
        """Handle scan initiation via Socket.IO."""
        global scan_in_progress
        if scan_in_progress:
            await sio.emit("error", {"message": "Scan already in progress"}, room=sid)
            return
        scan_in_progress = True
        try:
            params = ScanParams(**data)
            async with MultiPlatformAPI() as api:
                global scanner
                scanner = AdvancedForkScanner(api, db)
                results = await scanner.scan_forks(
                    days=params.days,
                    all_branches=params.all_branches,
                    max_forks=params.max_forks,
                    quiet=True
                )
                await sio.emit("stats", scanner.scan_stats)
                await sio.emit("scan_complete", {"results": [r.dict() for r in results]})
        except Exception as e:
            await sio.emit("error", {"message": str(e)}, room=sid)
        finally:
            scan_in_progress = False

# --- CLI Interface ---
app = Typer()

@app.command()
def scan(
    days: int = Option(365, help="Days to look back for fork activity"),
    all_branches: bool = Option(False, "--all-branches", help="Scan all branches"),
    max_forks: int = Option(100, help="Maximum forks to scan"),
    deep_scan: bool = Option(False, "--deep-scan", help="Use deep scan parameters")
):
    """Scan ScummVM forks for hidden engines."""
    if deep_scan:
        days = settings.deep_scan_days
    async def run_scan():
        async with MultiPlatformAPI() as api:
            scanner = AdvancedForkScanner(api, Database())
            await scanner.scan_forks(
                days=days,
                all_branches=all_branches,
                max_forks=max_forks,
                quiet=False
            )
    asyncio.run(run_scan())

@app.command()
def web(port: int = Option(settings.web_service_port, help="Port for web service")):
    """Start the web service."""
    if not WEB_SERVICE_AVAILABLE:
        console.print("Web service dependencies not available.")
        return
    config = Config(app=app, host="0.0.0.0", port=port, log_level="info")
    server = Server(config)
    asyncio.run(server.serve())

@app.command()
def train():
    """Train the ML model for engine classification."""
    logger.info("Training ML model...")
    async def run_training():
        async with MultiPlatformAPI() as api:
            analyzer = EngineAnalyzer(api)
            await analyzer.train_classifier()
    asyncio.run(run_training())
    console.print("ML model training completed.")

def display_results(results: List[ForkAnalysis], stats: Dict):
    """Display scan results and differential analysis."""
    table = Table(title="Scan Results", show_header=True)
    table.add_column("Fork", style="cyan")
    table.add_column("Engines", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("Classification", style="yellow")
    for fork in results:
        table.add_row(
            f"{fork.owner}/{fork.repo}",
            str(len(fork.engines)),
            str(fork.total_score),
            fork.classification
        )
    console.print(table)
    
    diff_table = Table(title="Differential Analysis", show_header=True)
    diff_table.add_column("Metric", style="blue")
    diff_table.add_column("Value", style="green")
    diff_table.add_row("New Forks", str(stats["new_forks"]))
    diff_table.add_row("Updated Forks", str(stats["updated_forks"]))
    diff_table.add_row("Removed Forks", str(stats["removed_forks"]))
    diff_table.add_row("New Engines", str(stats["new_engines"]))
    diff_table.add_row("Updated Engines", str(stats["updated_engines"]))
    diff_table.add_row("Removed Engines", str(stats["removed_engines"]))
    console.print(diff_table)
    
    console.print(f"\nScan Duration: {stats["duration"]:.2f} seconds")

if __name__ == "__main__":
    app()

