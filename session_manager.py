"""Conversation session management with in-memory storage."""
import time
from typing import Dict, List, Optional
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class Message:
    """Single conversation message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float


class SessionManager:
    """
    Manages conversation sessions with LRU eviction and TTL expiration.

    Features:
    - Automatic cleanup of expired sessions
    - LRU eviction when max sessions exceeded
    - Keep only last N turns per session
    """

    def __init__(self, max_sessions: int = 1000, ttl: int = 1800):
        """
        Args:
            max_sessions: Maximum number of concurrent sessions
            ttl: Time-to-live in seconds (default: 30 minutes)
        """
        self._sessions: OrderedDict[str, List[Message]] = OrderedDict()
        self._last_access: Dict[str, float] = {}
        self.max_sessions = max_sessions
        self.ttl = ttl

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to session history.

        Args:
            session_id: Unique session identifier
            role: 'user' or 'assistant'
            content: Message content
        """
        self._cleanup_expired()

        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append(
            Message(role=role, content=content, timestamp=time.time())
        )
        self._last_access[session_id] = time.time()

        # Keep only last 10 messages (5 turns: user + assistant)
        max_messages = 10
        if len(self._sessions[session_id]) > max_messages:
            self._sessions[session_id] = self._sessions[session_id][-max_messages:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of messages in format [{"role": "user", "content": "..."}, ...]
        """
        self._cleanup_expired()

        if session_id not in self._sessions:
            return []

        self._last_access[session_id] = time.time()

        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._sessions[session_id]
        ]

    def clear_session(self, session_id: str) -> None:
        """Clear specific session."""
        self._sessions.pop(session_id, None)
        self._last_access.pop(session_id, None)

    def get_active_sessions_count(self) -> int:
        """Get count of currently active sessions."""
        self._cleanup_expired()
        return len(self._sessions)

    def _cleanup_expired(self) -> None:
        """Remove expired sessions based on TTL."""
        now = time.time()
        expired = [
            sid for sid, last_time in self._last_access.items()
            if now - last_time > self.ttl
        ]

        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)

        # Enforce max sessions (LRU eviction)
        while len(self._sessions) > self.max_sessions:
            oldest = next(iter(self._sessions))
            self._sessions.pop(oldest)
            self._last_access.pop(oldest, None)
