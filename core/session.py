"""core/session.py -- Collaborative Session Sharing & Persistence for SatQuery AI.

This module provides session serialization, deserialization, validation,
comparison, forking, tagging, template functionality, discovery, filtering,
search, archival, and checkpoint/version history.

Session files are UNTRUSTED DATA - they must be validated on load and
never used to invoke arbitrary tools or access secrets.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from copy import deepcopy
from pathlib import Path

from core.evidence import UNAVAILABLE
from core.planner import ConversationState, ConversationTurn


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SESSION_SCHEMA = "satquery-session/2"
SESSION_SCHEMA_VERSION = 2

# Session directory (configurable via env var)
SESSION_DIR_ENV = "SATQUERY_SESSION_DIR"
DEFAULT_SESSION_DIR = Path.cwd() / "sessions"

# Checkpoint settings
MAX_CHECKPOINTS_PER_SESSION = 10
CHECKPOINT_DIR_NAME = "checkpoints"
ARCHIVE_DIR_NAME = "archive"

# Input bounds for metadata strings
MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 5000
MAX_TAG_LENGTH = 50
MAX_TAGS = 20
MAX_ANNOTATION_TEXT_LENGTH = 2000
MAX_ANNOTATION_AUTHOR_LENGTH = 100
MAX_ANNOTATIONS = 50
MAX_CHECKPOINT_LABEL_LENGTH = 100

# Allowed characters for safe filenames
SAFE_FILENAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename to prevent path traversal.

    Only allows alphanumeric, dash, underscore, and dot.
    Returns a safe version or raises ValueError if too dangerous.
    """
    if not name:
        raise ValueError("Empty filename")

    # Check for path traversal attempts
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError(f"Invalid filename: path traversal detected in '{name}'")

    # Check each character
    for ch in name:
        if ch not in SAFE_FILENAME_CHARS:
            raise ValueError(f"Invalid character '{ch}' in filename '{name}'")

    # Limit length
    if len(name) > 255:
        raise ValueError(f"Filename too long (max 255 chars): '{name}'")

    return name


def _atomic_write(filepath: Path, content: str, encoding: str = "utf-8") -> None:
    """Write content to a file atomically using a temporary file.

    This prevents corruption if the process crashes mid-write.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file in the same directory (same filesystem for atomic rename)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=filepath.parent,
        prefix=f".{filepath.name}.tmp.",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Atomic rename (on POSIX and Windows with replace)
        tmp_path.replace(filepath)
    except Exception:
        # Clean up temp file on failure
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _atomic_write_bytes(filepath: Path, content: bytes) -> None:
    """Write bytes to a file atomically using a temporary file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=filepath.parent,
        prefix=f".{filepath.name}.tmp.",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        tmp_path.replace(filepath)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _validate_session_id(session_id: str) -> str:
    """Validate and sanitize a session ID for filesystem use."""
    # Session IDs should be alphanumeric with optional underscore/hyphen
    if not session_id:
        raise ValueError("Empty session ID")

    # Only allow alphanumeric, underscore, and hyphen
    if not all(c.isalnum() or c in "_-" for c in session_id):
        raise ValueError(f"Invalid session ID: '{session_id}' (only alphanumeric, underscore, hyphen allowed)")

    if len(session_id) > 64:
        raise ValueError(f"Session ID too long: '{session_id}'")

    return session_id


def _validate_checkpoint_filename(filename: str) -> str:
    """Validate a checkpoint filename."""
    return _sanitize_filename(filename)


def _validate_archive_filename(filename: str) -> str:
    """Validate an archive filename."""
    return _sanitize_filename(filename)


# --------------------------------------------------------------------------- #
# Data Structures
# --------------------------------------------------------------------------- #

@dataclass
class SessionAnnotation:
    """A reviewer annotation on a session."""
    text: str
    author: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "author": self.author, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionAnnotation":
        return cls(
            text=data.get("text", ""),
            author=data.get("author", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class SessionMetadata:
    """Session metadata including title, description, tags, and provenance."""
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    annotations: List[SessionAnnotation] = field(default_factory=list)
    session_id: str = ""
    parent_session_id: str = ""
    forked_at: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "title": self.title,
        }
        # Only include non-default fields
        if self.description:
            result["description"] = self.description
        if self.tags:
            result["tags"] = list(self.tags)
        if self.annotations:
            result["annotations"] = [a.to_dict() for a in self.annotations]
        if self.session_id:
            result["session_id"] = self.session_id
        if self.parent_session_id:
            result["parent_session_id"] = self.parent_session_id
        if self.forked_at:
            result["forked_at"] = self.forked_at
        # Always include timestamps
        result["created_at"] = self.created_at
        result["updated_at"] = self.updated_at
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        return cls(
            title=data.get("title", ""),
            description=data.get("description", ""),
            tags=list(data.get("tags", [])),
            annotations=[SessionAnnotation.from_dict(a) for a in data.get("annotations", [])],
            session_id=data.get("session_id", ""),
            parent_session_id=data.get("parent_session_id", ""),
            forked_at=data.get("forked_at", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class Session:
    """A complete SatQuery analysis session for sharing and persistence."""

    def __init__(
        self,
        metadata: Optional[SessionMetadata] = None,
        conversation_state: Optional[Dict[str, Any]] = None,
        raster_context: Optional[Dict[str, Any]] = None,
        roi_context: Optional[Dict[str, Any]] = None,
        evidence_package: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        current_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.schema = SESSION_SCHEMA
        self.version = SESSION_SCHEMA_VERSION
        self.metadata = metadata or SessionMetadata()
        self.conversation_state = conversation_state
        self.raster_context = raster_context
        self.roi_context = roi_context
        self.evidence_package = evidence_package
        self.chat_history = chat_history or []
        self.current_arguments = current_arguments or {}
        self.has_raster_loaded = False
        self.has_roi_selected = False

    # ----------------------------------------------------------------------- #
    # Serialization
    # ----------------------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to a dictionary (JSON-serializable)."""
        result = {
            "schema": self.schema,
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "chat_history": self.chat_history,
            "current_arguments": self.current_arguments,
            "has_raster_loaded": self.has_raster_loaded,
            "has_roi_selected": self.has_roi_selected,
        }
        # Only include optional fields if they have meaningful values
        if self.conversation_state is not None:
            result["conversation_state"] = self.conversation_state
        if self.raster_context is not None:
            result["raster_context"] = self.raster_context
        if self.roi_context is not None:
            result["roi_context"] = self.roi_context
        if self.evidence_package is not None:
            result["evidence_package"] = self.evidence_package
        return result

    def to_json(self) -> str:
        """Serialize session to JSON string, sanitizing secrets."""
        data = self.to_dict()
        # Sanitize metadata fields for banned tokens
        banned = ["api key", "apikey", "access token", "sign up for",
                  "password", "secret", "credential", "bearer"]

        if data.get("metadata"):
            metadata = data["metadata"]
            # Sanitize title and description
            for field in ("title", "description"):
                if field in metadata:
                    text = str(metadata[field]).lower()
                    for token in banned:
                        if token in text:
                            metadata[field] = "[REDACTED]"
            # Sanitize annotations
            if metadata.get("annotations"):
                for ann in metadata["annotations"]:
                    text = ann.get("text", "").lower()
                    for token in banned:
                        if token in text:
                            ann["text"] = "[REDACTED]"
        return json.dumps(data, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> "Session":
        """Deserialize session from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from dictionary (validates schema/version)."""
        # Validate schema - support v1 for backward compatibility
        schema = data.get("schema")
        if schema not in ("satquery-session/1", SESSION_SCHEMA):
            raise ValueError(f"Unsupported session schema: {schema}. Expected {SESSION_SCHEMA} or satquery-session/1")

        is_v1 = schema == "satquery-session/1"

        # Validate version
        version = data.get("version", 0)
        if not isinstance(version, int) or version < 1:
            raise ValueError(f"Invalid session version: {version}")
        if version > SESSION_SCHEMA_VERSION:
            raise ValueError(f"Session version {version} is newer than supported ({SESSION_SCHEMA_VERSION})")

        # Create session
        session = cls()
        session.schema = SESSION_SCHEMA  # Always upgrade to v2
        session.version = SESSION_SCHEMA_VERSION
        session.metadata = SessionMetadata.from_dict(data.get("metadata", {}))
        session.conversation_state = data.get("conversation_state")
        session.raster_context = data.get("raster_context")
        session.roi_context = data.get("roi_context")
        session.evidence_package = data.get("evidence_package")
        session.chat_history = data.get("chat_history", [])
        session.current_arguments = data.get("current_arguments", {})
        session.has_raster_loaded = data.get("has_raster_loaded", False)
        session.has_roi_selected = data.get("has_roi_selected", False)

        # For v1 sessions, don't auto-generate session_id (preserve empty)
        if not is_v1 and not session.metadata.session_id:
            session.metadata.session_id = str(uuid.uuid4())[:8]

        return session

    # ----------------------------------------------------------------------- #
    # Validation
    # ----------------------------------------------------------------------- #

    def validate(self) -> tuple[bool, List[str]]:
        """Validate session structure. Returns (is_valid, errors)."""
        errors = []

        # Check required fields
        if not self.schema or self.schema != SESSION_SCHEMA:
            errors.append(f"Invalid schema: {self.schema}")

        if not isinstance(self.version, int) or self.version < 1:
            errors.append(f"Invalid version: {self.version}")

        # Validate metadata
        if self.metadata is None:
            errors.append("Missing metadata")
        else:
            # Metadata fields should be strings/lists
            if not isinstance(self.metadata.title, str):
                errors.append("metadata.title must be string")
            if not isinstance(self.metadata.description, str):
                errors.append("metadata.description must be string")
            if not isinstance(self.metadata.tags, list):
                errors.append("metadata.tags must be list")
            if not isinstance(self.metadata.annotations, list):
                errors.append("metadata.annotations must be list")

        # Validate conversation_state if present
        if self.conversation_state is not None:
            if not isinstance(self.conversation_state, dict):
                errors.append("conversation_state must be dict")
            else:
                required_cs = ["current_roi_available", "current_crop", "current_dates", "current_intent", "recent_turns"]
                for field_name in required_cs:
                    if field_name not in self.conversation_state:
                        errors.append(f"conversation_state missing required field: {field_name}")

        # Validate evidence_package if present
        if self.evidence_package is not None:
            if not isinstance(self.evidence_package, dict):
                errors.append("evidence_package must be dict")

        return len(errors) == 0, errors

    # ----------------------------------------------------------------------- #
    # Session Comparison (Step 10)
    # ----------------------------------------------------------------------- #

    def _diff_dicts(self, d1: Dict, d2: Dict, prefix: str = "") -> Dict[str, Dict]:
        """Recursively compare two dictionaries."""
        # Fields to exclude from diff (timestamps, etc.)
        EXCLUDE_FIELDS = {
            "metadata.created_at",
            "metadata.updated_at",
            "metadata.forked_at",
            "metadata.annotations.timestamp",
        }

        result = {"unchanged": {}, "added": {}, "removed": {}, "changed": {}, "unavailable": {}}
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            full_key = f"{prefix}{key}"
            if full_key in EXCLUDE_FIELDS:
                # Skip timestamp fields
                continue
            v1 = d1.get(key, UNAVAILABLE)
            v2 = d2.get(key, UNAVAILABLE)

            if v1 == UNAVAILABLE and v2 == UNAVAILABLE:
                result["unavailable"][full_key] = True
            elif v1 == UNAVAILABLE:
                result["added"][full_key] = v2
            elif v2 == UNAVAILABLE:
                result["removed"][full_key] = v1
            elif v1 == v2:
                result["unchanged"][full_key] = v1
            else:
                if isinstance(v1, dict) and isinstance(v2, dict):
                    nested = self._diff_dicts(v1, v2, f"{full_key}.")
                    for cat in ("unchanged", "added", "removed", "changed", "unavailable"):
                        result[cat].update(nested[cat])
                else:
                    result["changed"][full_key] = {"from": v1, "to": v2}

        return result

    def diff(self, other: "Session") -> Dict[str, Dict]:
        """Compare this session with another, returning detailed differences."""
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        return self._diff_dicts(self_dict, other_dict)

    def get_comparison_summary(self, other: "Session") -> Dict[str, Any]:
        """Get a human-readable summary of differences between sessions."""
        diff = self.diff(other)

        summary = {
            "metadata_changed": False,
            "conversation_changed": False,
            "evidence_changed": False,
            "raster_changed": False,
            "roi_changed": False,
            "tags_changed": False,
            "title_changed": False,
            "summary": {
                "added": len(diff["added"]),
                "removed": len(diff["removed"]),
                "changed": len(diff["changed"]),
                "unchanged": len(diff["unchanged"]),
            }
        }

        # Check specific important fields
        for key in ("metadata", "metadata.title", "metadata.tags", "metadata.description"):
            if key in diff["changed"] or key in diff["added"] or key in diff["removed"]:
                summary["metadata_changed"] = True
        if "metadata.title" in diff["changed"] or "metadata.title" in diff["added"] or "metadata.title" in diff["removed"]:
            summary["title_changed"] = True
        if "metadata.tags" in diff["changed"] or "metadata.tags" in diff["added"] or "metadata.tags" in diff["removed"]:
            summary["tags_changed"] = True

        if "conversation_state" in diff["changed"] or "conversation_state" in diff["added"] or "conversation_state" in diff["removed"]:
            summary["conversation_changed"] = True
        if "evidence_package" in diff["changed"] or "evidence_package" in diff["added"] or "evidence_package" in diff["removed"]:
            summary["evidence_changed"] = True
        if "raster_context" in diff["changed"] or "raster_context" in diff["added"] or "raster_context" in diff["removed"]:
            summary["raster_changed"] = True
        if "roi_context" in diff["changed"] or "roi_context" in diff["added"] or "roi_context" in diff["removed"]:
            summary["roi_changed"] = True

        return summary

    # ----------------------------------------------------------------------- #
    # Session Forking (Step 10)
    # ----------------------------------------------------------------------- #

    def fork(self, new_title: str = "") -> "Session":
        """Create an independent fork of this session.

        The fork preserves all conversation history, evidence, and metadata,
        but gets a new session_id and records the parent relationship.
        """
        # Ensure parent has a session_id
        parent_id = self.session_id

        forked = deepcopy(self)

        # Generate new session ID
        forked.metadata.session_id = str(uuid.uuid4())[:8]
        forked.metadata.parent_session_id = parent_id
        forked.metadata.forked_at = datetime.now().isoformat()
        forked.metadata.updated_at = datetime.now().isoformat()

        # Reset tags and annotations - fork starts fresh
        forked.metadata.tags = []
        forked.metadata.annotations = []

        # Update title if provided (non-empty and not just "Fork")
        if new_title and new_title != "Fork":
            forked.metadata.title = new_title
        elif not forked.metadata.title:
            forked.metadata.title = f"Fork of {self.metadata.title or 'Untitled Session'}"

        # Reset authoritative flags - fork doesn't inherit authoritative raster/ROI
        forked.has_raster_loaded = False
        forked.has_roi_selected = False

        return forked

    # ----------------------------------------------------------------------- #
    # Session Templates (Step 10)
    # ----------------------------------------------------------------------- #

    @classmethod
    def from_template(cls, template_name: str, title: str = "") -> "Session":
        """Create a new session from a workflow template."""
        templates = cls.get_templates()

        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")

        template = templates[template_name]

        session = cls()
        session.metadata.title = title or template.get("title", template_name)
        session.metadata.description = template.get("description", "")
        session.metadata.tags = template.get("tags", [])
        session.metadata.session_id = str(uuid.uuid4())[:8]
        session.metadata.created_at = datetime.now().isoformat()
        session.metadata.updated_at = datetime.now().isoformat()

        return session

    @classmethod
    def get_templates(cls) -> Dict[str, Dict[str, Any]]:
        """Get available workflow templates."""
        return {
            "ndvi_monitoring": {
                "title": "NDVI Monitoring",
                "description": "Monitor vegetation health over time using NDVI",
                "tags": ["ndvi", "monitoring", "vegetation"],
                "expected_intent": "NDVI_ROI_STATS",
                "required_inputs": ["roi"],
                "optional_inputs": ["dates"],
            },
            "ndvi_change_detection": {
                "title": "NDVI Change Detection",
                "description": "Detect vegetation changes between two dates",
                "tags": ["ndvi", "change", "temporal", "comparison"],
                "expected_intent": "NDVI_CHANGE_ROI",
                "required_inputs": ["roi", "date1", "date2"],
                "optional_inputs": [],
            },
            "ndwi_water_analysis": {
                "title": "NDWI Water Analysis",
                "description": "Analyze water content and water bodies using NDWI",
                "tags": ["ndwi", "water", "analysis"],
                "expected_intent": "NDWI_ROI_STATS",
                "required_inputs": ["roi"],
                "optional_inputs": [],
            },
            "crop_suitability": {
                "title": "Crop Suitability Assessment",
                "description": "Assess land suitability for specific crops",
                "tags": ["agriculture", "crop", "suitability", "cotton"],
                "expected_intent": "CROP_SUITABILITY",
                "required_inputs": ["roi", "crop"],
                "optional_inputs": [],
            },
            "multi_condition_spatial": {
                "title": "Multi-Condition Spatial Query",
                "description": "Combine multiple spatial conditions (land cover, indices, distance)",
                "tags": ["spatial", "multi-condition", "composite"],
                "expected_intent": "MULTI_CONDITION",
                "required_inputs": ["roi", "conditions"],
                "optional_inputs": ["dates"],
            },
            "temporal_ndvi_comparison": {
                "title": "Temporal NDVI Comparison",
                "description": "Compare NDVI statistics across multiple time periods",
                "tags": ["ndvi", "temporal", "statistics", "comparison"],
                "expected_intent": "TEMPORAL_COMPARISON",
                "required_inputs": ["roi", "dates"],
                "optional_inputs": [],
            },
        }

    # ----------------------------------------------------------------------- #
    # Metadata Helpers (Step 10)
    # ----------------------------------------------------------------------- #

    def _validate_title(self, title: str) -> None:
        if len(title) > MAX_TITLE_LENGTH:
            raise ValueError(f"Title too long (max {MAX_TITLE_LENGTH} chars)")

    def _validate_description(self, description: str) -> None:
        if len(description) > MAX_DESCRIPTION_LENGTH:
            raise ValueError(f"Description too long (max {MAX_DESCRIPTION_LENGTH} chars)")

    def _validate_tag(self, tag: str) -> None:
        if len(tag) > MAX_TAG_LENGTH:
            raise ValueError(f"Tag too long (max {MAX_TAG_LENGTH} chars)")
        if len(self.metadata.tags) >= MAX_TAGS and tag not in self.metadata.tags:
            raise ValueError(f"Too many tags (max {MAX_TAGS})")

    def _validate_annotation(self, text: str, author: str) -> None:
        if len(text) > MAX_ANNOTATION_TEXT_LENGTH:
            raise ValueError(f"Annotation text too long (max {MAX_ANNOTATION_TEXT_LENGTH} chars)")
        if len(author) > MAX_ANNOTATION_AUTHOR_LENGTH:
            raise ValueError(f"Annotation author too long (max {MAX_ANNOTATION_AUTHOR_LENGTH} chars)")
        if len(self.metadata.annotations) >= MAX_ANNOTATIONS:
            raise ValueError(f"Too many annotations (max {MAX_ANNOTATIONS})")

    def update_metadata(self, title: Optional[str] = None, description: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> None:
        """Update session metadata fields."""
        if title is not None:
            self._validate_title(title)
            self.metadata.title = title
        if description is not None:
            self._validate_description(description)
            self.metadata.description = description
        if tags is not None:
            for tag in tags:
                self._validate_tag(tag)
            self.metadata.tags = tags
        self.metadata.updated_at = datetime.now().isoformat()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the session."""
        self._validate_tag(tag)
        if tag not in self.metadata.tags:
            self.metadata.tags.append(tag)
            self.metadata.updated_at = datetime.now().isoformat()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the session."""
        if tag in self.metadata.tags:
            self.metadata.tags.remove(tag)
            self.metadata.updated_at = datetime.now().isoformat()

    def add_annotation(self, text: str, author: str = "") -> None:
        """Add a reviewer annotation."""
        self._validate_annotation(text, author)
        annotation = SessionAnnotation(
            text=text,
            author=author,
            timestamp=datetime.now().isoformat(),
        )
        self.metadata.annotations.append(annotation)
        self.metadata.updated_at = datetime.now().isoformat()

    # ----------------------------------------------------------------------- #
    # Session ID Property
    # ----------------------------------------------------------------------- #

    @property
    def session_id(self) -> str:
        """Get the session ID, generating one if not present."""
        if not self.metadata.session_id:
            self.metadata.session_id = str(uuid.uuid4())[:8]
        return self.metadata.session_id

    @property
    def short_id(self) -> str:
        """Get a short display ID."""
        return self.session_id[:8] if self.session_id else "new"


# --------------------------------------------------------------------------- #
# Session I/O Helpers
# --------------------------------------------------------------------------- #

def build_session_from_app_state(
    conversation_state: Optional[ConversationState] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    current_raster: Optional[Any] = None,
    current_roi: Optional[Any] = None,
    current_arguments: Optional[Dict[str, Any]] = None,
    evidence_package: Optional[Dict[str, Any]] = None,
) -> Session:
    """Build a Session from current application state."""
    # Convert conversation state to dict
    cs_dict = None
    if conversation_state is not None:
        cs_dict = conversation_state.get_context_summary()

    # Build raster context (only metadata, no arrays)
    raster_ctx = None
    if current_raster is not None:
        raster_ctx = {
            "path": getattr(current_raster, "path", None),
            "label": getattr(current_raster, "label", None),
            "crs": getattr(current_raster, "crs", None),
        }

    # Build ROI context (only metadata, no geometry)
    roi_ctx = None
    if current_roi is not None and getattr(current_roi, "usable", False):
        roi_ctx = {
            "area_m2": getattr(current_roi, "area_m2", None),
            "crs": getattr(current_roi, "raster_crs", None),
        }

    session = Session(
        conversation_state=cs_dict,
        raster_context=raster_ctx,
        roi_context=roi_ctx,
        evidence_package=evidence_package,
        chat_history=chat_history or [],
        current_arguments=current_arguments or {},
    )

    # Set authoritative flags
    session.has_raster_loaded = current_raster is not None
    session.has_roi_selected = current_roi is not None and getattr(current_roi, "usable", False)

    return session


def save_session_to_file(session: Session, filepath: str) -> None:
    """Save session to a JSON file atomically."""
    _atomic_write(Path(filepath), session.to_json())


def load_session_from_file(filepath: str) -> Session:
    """Load session from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        json_str = f.read()
    return Session.from_json(json_str)


def apply_session_to_conversation_state(session: Session, conversation_state: ConversationState) -> None:
    """Apply loaded session data to a ConversationState object."""
    if session.conversation_state is None:
        return

    cs = session.conversation_state

    # Restore recent turns
    for turn_data in cs.get("recent_turns", []):
        turn = ConversationTurn(
            user_query=turn_data.get("query", ""),
            tool_name=turn_data.get("tool"),
            intent=turn_data.get("intent"),
            status=turn_data.get("status"),
            crop=turn_data.get("crop"),
            has_roi=turn_data.get("has_roi", False),
            dates=tuple(turn_data.get("dates", (None, None))) if turn_data.get("dates") else (None, None),
        )
        conversation_state.add_turn(turn)

    # Restore current context flags
    conversation_state.current_roi_available = cs.get("current_roi_available", False)
    conversation_state.current_crop = cs.get("current_crop")
    conversation_state.current_dates = tuple(cs.get("current_dates", (None, None)))
    conversation_state.current_intent = cs.get("current_intent")


def get_session_summary(session: Session) -> Dict[str, Any]:
    """Get a compact summary of the session for display."""
    return {
        "session_id": session.session_id,
        "title": session.metadata.title,
        "description": session.metadata.description,
        "tags": session.metadata.tags,
        "created_at": session.metadata.created_at,
        "updated_at": session.metadata.updated_at,
        "parent_session_id": session.metadata.parent_session_id,
        "forked_at": session.metadata.forked_at,
        "has_conversation": session.conversation_state is not None,
        "has_evidence": session.evidence_package is not None,
        "has_raster": session.has_raster_loaded,
        "has_roi": session.has_roi_selected,
        "num_annotations": len(session.metadata.annotations),
        "num_chat_entries": len(session.chat_history),
    }


def _get_session_dir() -> Path:
    """Get the session directory, creating it if needed."""
    env_dir = os.environ.get(SESSION_DIR_ENV)
    if env_dir:
        session_dir = Path(env_dir)
    else:
        session_dir = DEFAULT_SESSION_DIR
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _get_checkpoint_dir(session_id: str) -> Path:
    """Get the checkpoint directory for a session."""
    session_dir = _get_session_dir()
    checkpoint_dir = session_dir / CHECKPOINT_DIR_NAME / session_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _get_archive_dir() -> Path:
    """Get the archive directory."""
    session_dir = _get_session_dir()
    archive_dir = session_dir / ARCHIVE_DIR_NAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def _list_session_files() -> List[Path]:
    """List all session JSON files in the session directory (non-recursive)."""
    session_dir = _get_session_dir()
    return sorted(session_dir.glob("*.json"))


def _parse_datetime_safe(dt_str: str) -> Optional[datetime]:
    """Parse ISO datetime string safely, returning None on failure."""
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


# --------------------------------------------------------------------------- #
# Session Discovery & Listing (Step 11)
# --------------------------------------------------------------------------- #

@dataclass
class SessionListEntry:
    """Lightweight session entry for listing/browsing."""
    session_id: str
    title: str
    description: str
    tags: List[str]
    created_at: str
    updated_at: str
    parent_session_id: str
    forked_at: str
    has_conversation: bool
    has_evidence: bool
    has_raster: bool
    has_roi: bool
    num_annotations: int
    num_chat_entries: int
    filepath: str
    intent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_session_id": self.parent_session_id,
            "forked_at": self.forked_at,
            "has_conversation": self.has_conversation,
            "has_evidence": self.has_evidence,
            "has_raster": self.has_raster,
            "has_roi": self.has_roi,
            "num_annotations": self.num_annotations,
            "num_chat_entries": self.num_chat_entries,
            "filepath": self.filepath,
            "intent": self.intent,
        }


def _extract_intent_from_session(session: Session) -> Optional[str]:
    """Extract the most recent intent from a session."""
    if session.conversation_state:
        return session.conversation_state.get("current_intent")
    # Fallback: check chat history
    if session.chat_history:
        for entry in reversed(session.chat_history):
            if entry.get("intent"):
                return entry["intent"]
    return None


def load_session_list_entry(filepath: Path) -> Optional[SessionListEntry]:
    """Load a session file and return a lightweight list entry.

    Handles missing/malformed files safely by returning None.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            json_str = f.read()
        session = Session.from_json(json_str)
        is_valid, errors = session.validate()
        if not is_valid:
            # Still return entry but mark as invalid
            return SessionListEntry(
                session_id=session.session_id or filepath.stem,
                title=session.metadata.title or "(invalid)",
                description=session.metadata.description,
                tags=session.metadata.tags,
                created_at=session.metadata.created_at or "",
                updated_at=session.metadata.updated_at or "",
                parent_session_id=session.metadata.parent_session_id or "",
                forked_at=session.metadata.forked_at or "",
                has_conversation=False,
                has_evidence=False,
                has_raster=False,
                has_roi=False,
                num_annotations=0,
                num_chat_entries=0,
                filepath=str(filepath),
                intent=None,
            )

        return SessionListEntry(
            session_id=session.session_id,
            title=session.metadata.title,
            description=session.metadata.description,
            tags=list(session.metadata.tags),
            created_at=session.metadata.created_at,
            updated_at=session.metadata.updated_at,
            parent_session_id=session.metadata.parent_session_id,
            forked_at=session.metadata.forked_at,
            has_conversation=session.conversation_state is not None,
            has_evidence=session.evidence_package is not None,
            has_raster=session.has_raster_loaded,
            has_roi=session.has_roi_selected,
            num_annotations=len(session.metadata.annotations),
            num_chat_entries=len(session.chat_history),
            filepath=str(filepath),
            intent=_extract_intent_from_session(session),
        )
    except (json.JSONDecodeError, ValueError, OSError, UnicodeDecodeError):
        # Malformed or unreadable file - return minimal entry
        return SessionListEntry(
            session_id=filepath.stem,
            title="(unreadable)",
            description="",
            tags=[],
            created_at="",
            updated_at="",
            parent_session_id="",
            forked_at="",
            has_conversation=False,
            has_evidence=False,
            has_raster=False,
            has_roi=False,
            num_annotations=0,
            num_chat_entries=0,
            filepath=str(filepath),
            intent=None,
        )


def list_sessions() -> List[SessionListEntry]:
    """List all available session files with metadata.

    Returns a list of SessionListEntry objects sorted by updated_at (newest first).
    Invalid/malformed files are included but marked accordingly.
    """
    entries = []
    for filepath in _list_session_files():
        entry = load_session_list_entry(filepath)
        if entry:
            entries.append(entry)

    # Sort by updated_at descending (newest first), with invalid/empty dates at end
    def sort_key(e: SessionListEntry):
        dt = _parse_datetime_safe(e.updated_at)
        return dt if dt else datetime.min

    entries.sort(key=sort_key, reverse=True)
    return entries


# --------------------------------------------------------------------------- #
# Session Filtering (Step 11)
# --------------------------------------------------------------------------- #

@dataclass
class SessionFilter:
    """Filter criteria for session listing."""
    tags: Optional[List[str]] = None          # Must have ALL specified tags
    tag_any: Optional[List[str]] = None       # Must have ANY of these tags
    date_from: Optional[str] = None           # ISO date string (inclusive)
    date_to: Optional[str] = None             # ISO date string (inclusive)
    intent: Optional[str] = None              # Filter by intent
    has_evidence: Optional[bool] = None       # Filter by evidence presence
    has_conversation: Optional[bool] = None   # Filter by conversation presence

    def matches(self, entry: SessionListEntry) -> bool:
        """Check if an entry matches this filter."""
        # Tag filtering (ALL tags must match)
        if self.tags:
            if not all(tag in entry.tags for tag in self.tags):
                return False

        # Tag filtering (ANY tag must match)
        if self.tag_any:
            if not any(tag in entry.tags for tag in self.tag_any):
                return False

        # Date filtering
        if self.date_from:
            entry_dt = _parse_datetime_safe(entry.updated_at)
            filter_dt = _parse_datetime_safe(self.date_from)
            if entry_dt and filter_dt and entry_dt < filter_dt:
                return False

        if self.date_to:
            entry_dt = _parse_datetime_safe(entry.updated_at)
            filter_dt = _parse_datetime_safe(self.date_to)
            if entry_dt and filter_dt and entry_dt > filter_dt:
                return False

        # Intent filtering (case-insensitive)
        if self.intent:
            if not entry.intent or self.intent.lower() != entry.intent.lower():
                return False

        # Evidence filtering
        if self.has_evidence is not None:
            if entry.has_evidence != self.has_evidence:
                return False

        # Conversation filtering
        if self.has_conversation is not None:
            if entry.has_conversation != self.has_conversation:
                return False

        return True


def filter_sessions(entries: List[SessionListEntry], filter_criteria: SessionFilter) -> List[SessionListEntry]:
    """Filter a list of sessions by the given criteria."""
    return [e for e in entries if filter_criteria.matches(e)]


# --------------------------------------------------------------------------- #
# Session Metadata Search (Step 11)
# --------------------------------------------------------------------------- #

def search_sessions(entries: List[SessionListEntry], query: str) -> List[SessionListEntry]:
    """Search sessions by metadata (title, description, tags).

    Search is deterministic and case-insensitive. Does not search raw
    raster data or evidence arrays.
    """
    if not query or not query.strip():
        return entries

    query_lower = query.strip().lower()
    results = []

    for entry in entries:
        # Search in title
        if query_lower in entry.title.lower():
            results.append(entry)
            continue

        # Search in description
        if query_lower in entry.description.lower():
            results.append(entry)
            continue

        # Search in tags
        if any(query_lower in tag.lower() for tag in entry.tags):
            results.append(entry)
            continue

        # Search in session_id
        if query_lower in entry.session_id.lower():
            results.append(entry)
            continue

        # Search in parent_session_id
        if query_lower in entry.parent_session_id.lower():
            results.append(entry)
            continue

    return results


# --------------------------------------------------------------------------- #
# Session Archival/Cleanup (Step 11)
# --------------------------------------------------------------------------- #

def archive_session(session_id: str) -> tuple[bool, str]:
    """Archive a session by moving it to the archive directory.

    This is a safe operation - the session is moved, not deleted.
    Returns (success, message).
    """
    # Validate session_id
    try:
        session_id = _validate_session_id(session_id)
    except ValueError as e:
        return False, f"Invalid session ID: {e}"

    session_dir = _get_session_dir()
    archive_dir = _get_archive_dir()

    # Find the session file
    session_file = None
    for f in session_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                session = Session.from_json(fp.read())
                if session.session_id == session_id or f.stem == session_id:
                    session_file = f
                    break
        except Exception:
            continue

    if not session_file:
        return False, f"Session {session_id} not found"

    # Move to archive with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{session_file.stem}_archived_{timestamp}.json"

    # Validate archive filename
    try:
        archive_name = _validate_archive_filename(archive_name)
    except ValueError as e:
        return False, f"Invalid archive filename: {e}"

    archive_path = archive_dir / archive_name

    try:
        # Use atomic move via copy + unlink for safety
        shutil.copy2(str(session_file), str(archive_path))
        session_file.unlink()
        return True, f"Session archived to {archive_path.name}"
    except OSError as e:
        # Clean up on failure
        try:
            archive_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False, f"Failed to archive session: {e}"


def delete_session(session_id: str, confirm: bool = False) -> tuple[bool, str]:
    """Explicitly delete a session file.

    Requires confirm=True to prevent accidental deletion.
    Returns (success, message).
    """
    if not confirm:
        return False, "Deletion requires explicit confirmation (confirm=True)"

    # Validate session_id
    try:
        session_id = _validate_session_id(session_id)
    except ValueError as e:
        return False, f"Invalid session ID: {e}"

    session_dir = _get_session_dir()

    # Find the session file
    session_file = None
    for f in session_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                session = Session.from_json(fp.read())
                if session.session_id == session_id or f.stem == session_id:
                    session_file = f
                    break
        except Exception:
            continue

    if not session_file:
        return False, f"Session {session_id} not found"

    try:
        session_file.unlink()
        # Also clean up checkpoints
        checkpoint_dir = _get_checkpoint_dir(session_id)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        return True, f"Session {session_id} deleted"
    except OSError as e:
        return False, f"Failed to delete session: {e}"


def list_archived_sessions() -> List[SessionListEntry]:
    """List all archived sessions."""
    archive_dir = _get_archive_dir()
    entries = []
    for filepath in sorted(archive_dir.glob("*.json")):
        entry = load_session_list_entry(filepath)
        if entry:
            entries.append(entry)
    return entries


def restore_archived_session(archive_filename: str) -> tuple[bool, str]:
    """Restore an archived session back to the main session directory."""
    # Validate archive filename
    try:
        archive_filename = _validate_archive_filename(archive_filename)
    except ValueError as e:
        return False, f"Invalid archive filename: {e}"

    archive_dir = _get_archive_dir()
    archive_path = archive_dir / archive_filename

    if not archive_path.exists():
        return False, f"Archived session {archive_filename} not found"

    session_dir = _get_session_dir()
    # Remove "_archived_<timestamp>" suffix to get original name
    original_name = archive_filename.replace("_archived_", "_").split("_archived_")[0] + ".json"
    # If the name still has archive pattern, just use the stem without timestamp
    if "_archived_" in original_name:
        original_name = original_name.split("_archived_")[0] + ".json"

    # Validate the restore filename
    try:
        original_name = _sanitize_filename(original_name)
    except ValueError as e:
        return False, f"Invalid restore filename: {e}"

    restore_path = session_dir / original_name

    # Avoid overwrite
    counter = 1
    while restore_path.exists():
        stem = restore_path.stem
        restore_path = session_dir / f"{stem}_restored_{counter}.json"
        counter += 1

    try:
        shutil.copy2(str(archive_path), str(restore_path))
        return True, f"Session restored as {restore_path.name}"
    except OSError as e:
        # Clean up on failure
        try:
            restore_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False, f"Failed to restore session: {e}"


# --------------------------------------------------------------------------- #
# Session Checkpoints / Version History (Step 11)
# --------------------------------------------------------------------------- #

def create_checkpoint(session: Session, label: str = "") -> tuple[bool, str]:
    """Create a lightweight checkpoint of the current session state.

    Checkpoints are stored in a bounded history (max 10 per session).
    They do NOT contain raw raster arrays, secrets, credentials, or executable content.
    Returns (success, checkpoint_path_or_error_message).
    """
    # Validate session_id
    try:
        session_id = _validate_session_id(session.session_id)
    except ValueError as e:
        return False, f"Invalid session ID: {e}"

    # Validate label
    if label:
        if len(label) > MAX_CHECKPOINT_LABEL_LENGTH:
            return False, f"Checkpoint label too long (max {MAX_CHECKPOINT_LABEL_LENGTH} chars)"
        # Only allow safe characters in label
        for ch in label:
            if ch not in SAFE_FILENAME_CHARS:
                return False, f"Invalid character '{ch}' in checkpoint label"

    checkpoint_dir = _get_checkpoint_dir(session_id)

    # Create checkpoint data (lightweight - no raster arrays, no secrets)
    checkpoint_data = {
        "schema": SESSION_SCHEMA,
        "version": SESSION_SCHEMA_VERSION,
        "checkpoint": {
            "session_id": session_id,
            "label": label,
            "created_at": datetime.now().isoformat(),
            "parent_checkpoint": None,  # Could be extended for branching
        },
        "metadata": session.metadata.to_dict(),
        "conversation_state": session.conversation_state,
        "chat_history": session.chat_history,
        "current_arguments": session.current_arguments,
        # Intentionally exclude: raster_context, roi_context, evidence_package
        # These are large and not needed for checkpoint rollback
    }

    # Generate checkpoint filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_suffix = f"_{label}" if label else ""
    checkpoint_filename = f"checkpoint_{timestamp}{label_suffix}.json"

    # Validate checkpoint filename
    try:
        checkpoint_filename = _validate_checkpoint_filename(checkpoint_filename)
    except ValueError as e:
        return False, f"Invalid checkpoint filename: {e}"

    checkpoint_path = checkpoint_dir / checkpoint_filename

    try:
        _atomic_write(checkpoint_path, json.dumps(checkpoint_data, indent=2, sort_keys=True))
    except OSError as e:
        return False, f"Failed to write checkpoint: {e}"

    # Enforce bounded history - remove oldest checkpoints if over limit
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))
    while len(checkpoints) > MAX_CHECKPOINTS_PER_SESSION:
        try:
            checkpoints[0].unlink()
        except OSError:
            pass
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))

    return True, str(checkpoint_path)


def list_checkpoints(session_id: str) -> List[Dict[str, Any]]:
    """List all checkpoints for a session, newest first."""
    checkpoint_dir = _get_checkpoint_dir(session_id)
    checkpoints = []

    for filepath in sorted(checkpoint_dir.glob("checkpoint_*.json"), reverse=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            cp = data.get("checkpoint", {})
            checkpoints.append({
                "filename": filepath.name,
                "session_id": cp.get("session_id", session_id),
                "label": cp.get("label", ""),
                "created_at": cp.get("created_at", ""),
                "filepath": str(filepath),
            })
        except (json.JSONDecodeError, OSError):
            # Skip unreadable checkpoints
            continue

    return checkpoints


def load_checkpoint(session_id: str, checkpoint_filename: str) -> Optional[Session]:
    """Load a session from a checkpoint.

    Returns a Session object with checkpoint data. Note that raster/ROI
    context and evidence are not restored from checkpoints - they must
    be re-selected by the user.
    """
    # Validate session_id
    try:
        session_id = _validate_session_id(session_id)
    except ValueError:
        return None

    # Validate checkpoint filename
    try:
        checkpoint_filename = _validate_checkpoint_filename(checkpoint_filename)
    except ValueError:
        return None

    checkpoint_dir = _get_checkpoint_dir(session_id)
    checkpoint_path = checkpoint_dir / checkpoint_filename

    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct session from checkpoint data
        session = Session()
        session.schema = data.get("schema", SESSION_SCHEMA)
        session.version = data.get("version", SESSION_SCHEMA_VERSION)
        session.metadata = SessionMetadata.from_dict(data.get("metadata", {}))
        session.conversation_state = data.get("conversation_state")
        session.chat_history = data.get("chat_history", [])
        session.current_arguments = data.get("current_arguments", {})
        session.has_raster_loaded = False  # Checkpoints don't preserve authoritative raster/ROI
        session.has_roi_selected = False

        return session
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def delete_checkpoint(session_id: str, checkpoint_filename: str) -> tuple[bool, str]:
    """Delete a specific checkpoint."""
    # Validate session_id
    try:
        session_id = _validate_session_id(session_id)
    except ValueError as e:
        return False, f"Invalid session ID: {e}"

    # Validate checkpoint filename
    try:
        checkpoint_filename = _validate_checkpoint_filename(checkpoint_filename)
    except ValueError as e:
        return False, f"Invalid checkpoint filename: {e}"

    checkpoint_dir = _get_checkpoint_dir(session_id)
    checkpoint_path = checkpoint_dir / checkpoint_filename

    if not checkpoint_path.exists():
        return False, f"Checkpoint {checkpoint_filename} not found"

    try:
        checkpoint_path.unlink()
        return True, f"Checkpoint deleted"
    except OSError as e:
        return False, f"Failed to delete checkpoint: {e}"


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #

__all__ = [
    "SESSION_SCHEMA",
    "SESSION_SCHEMA_VERSION",
    "Session",
    "SessionMetadata",
    "SessionAnnotation",
    "build_session_from_app_state",
    "save_session_to_file",
    "load_session_from_file",
    "apply_session_to_conversation_state",
    "get_session_summary",
    # Step 11: Session Organization & Discovery
    "SessionListEntry",
    "SessionFilter",
    "list_sessions",
    "filter_sessions",
    "search_sessions",
    "archive_session",
    "delete_session",
    "list_archived_sessions",
    "restore_archived_session",
    "create_checkpoint",
    "list_checkpoints",
    "load_checkpoint",
    "delete_checkpoint",
]
