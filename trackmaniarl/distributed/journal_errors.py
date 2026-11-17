"""Exceptions shared by the rollout journal components."""


class JournalPayloadConflictError(ValueError):
    """A rollout sequence was reused for different bytes."""


class JournalSchemaError(ValueError):
    """The journal does not implement the current durable schema."""
