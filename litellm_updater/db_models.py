"""SQLAlchemy ORM models for database persistence."""
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Provider(Base):
    """Provider (source) configuration."""

    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    base_url: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    prefix: Mapped[str | None] = mapped_column(String, nullable=True)
    default_ollama_mode: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    models: Mapped[list["Model"]] = relationship(
        "Model", back_populates="provider", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("type IN ('ollama', 'litellm')", name="check_provider_type"),
        CheckConstraint(
            "default_ollama_mode IS NULL OR default_ollama_mode IN ('ollama', 'openai')",
            name="check_default_ollama_mode",
        ),
    )


class Model(Base):
    """Model metadata and configuration."""

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_type: Mapped[str | None] = mapped_column(String, nullable=True)

    # Token limits
    context_window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # JSON fields
    capabilities: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    litellm_params: Mapped[str] = mapped_column(Text, nullable=False)  # JSON object
    raw_metadata: Mapped[str] = mapped_column(Text, nullable=False)  # JSON object
    user_params: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object

    # Ollama-specific
    ollama_mode: Mapped[str | None] = mapped_column(String, nullable=True)

    # Tracking
    first_seen: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    is_orphaned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    orphaned_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    user_modified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    provider: Mapped["Provider"] = relationship("Provider", back_populates="models")

    __table_args__ = (
        CheckConstraint(
            "ollama_mode IS NULL OR ollama_mode IN ('ollama', 'openai')",
            name="check_model_ollama_mode",
        ),
    )

    @property
    def capabilities_list(self) -> list[str]:
        """Parse capabilities JSON to list."""
        if not self.capabilities:
            return []
        return json.loads(self.capabilities)

    @capabilities_list.setter
    def capabilities_list(self, value: list[str]) -> None:
        """Store capabilities as JSON."""
        self.capabilities = json.dumps(value) if value else None

    @property
    def litellm_params_dict(self) -> dict[str, Any]:
        """Parse litellm_params JSON to dict."""
        return json.loads(self.litellm_params)

    @litellm_params_dict.setter
    def litellm_params_dict(self, value: dict[str, Any]) -> None:
        """Store litellm_params as JSON."""
        self.litellm_params = json.dumps(value)

    @property
    def raw_metadata_dict(self) -> dict[str, Any]:
        """Parse raw_metadata JSON to dict."""
        return json.loads(self.raw_metadata)

    @raw_metadata_dict.setter
    def raw_metadata_dict(self, value: dict[str, Any]) -> None:
        """Store raw_metadata as JSON."""
        self.raw_metadata = json.dumps(value)

    @property
    def user_params_dict(self) -> dict[str, Any] | None:
        """Parse user_params JSON to dict."""
        if not self.user_params:
            return None
        return json.loads(self.user_params)

    @user_params_dict.setter
    def user_params_dict(self, value: dict[str, Any] | None) -> None:
        """Store user_params as JSON."""
        self.user_params = json.dumps(value) if value else None

    @property
    def effective_params(self) -> dict[str, Any]:
        """Return user_params if available, else litellm_params."""
        return self.user_params_dict or self.litellm_params_dict

    def get_display_name(self, apply_prefix: bool = True) -> str:
        """Return model name with optional provider prefix."""
        if apply_prefix and self.provider and self.provider.prefix:
            return f"{self.provider.prefix}/{self.model_id}"
        return self.model_id
