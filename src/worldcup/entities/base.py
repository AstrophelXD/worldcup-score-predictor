from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SourceMetadata(BaseModel):
    """Traceability fields required on all curated records."""

    model_config = ConfigDict(extra="forbid")

    source_system: str
    source_record_id: str | None = None
    ingested_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class ManualOverrideMetadata(BaseModel):
    """Audit fields for manually edited records."""

    model_config = ConfigDict(extra="forbid")

    is_manual_override: bool = False
    override_reason: str | None = None
    override_by: str | None = None
    override_at: datetime | None = None
