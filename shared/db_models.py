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
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    access_groups: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    pricing_profile: Mapped[str | None] = mapped_column(String, nullable=True)
    pricing_override: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object
    sync_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    auto_detect_fim: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    model_filter: Mapped[str | None] = mapped_column(String, nullable=True)  # Regex or substring filter for model names
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
        CheckConstraint(
            "type IN ('ollama', 'openai', 'compat')",
            name="check_provider_type",
        ),
        CheckConstraint(
            "default_ollama_mode IS NULL OR default_ollama_mode IN ('ollama', 'ollama_chat', 'openai')",
            name="check_default_ollama_mode",
        ),
    )

    @property
    def tags_list(self) -> list[str]:
        """Parse provider tags JSON to list."""
        if not self.tags:
            return []
        return json.loads(self.tags)

    @tags_list.setter
    def tags_list(self, value: list[str]) -> None:
        """Store provider tags as JSON."""
        self.tags = json.dumps(value) if value else None

    @property
    def access_groups_list(self) -> list[str]:
        """Parse provider access_groups JSON to list."""
        if not self.access_groups:
            return []
        return json.loads(self.access_groups)

    @access_groups_list.setter
    def access_groups_list(self, value: list[str]) -> None:
        """Store provider access_groups as JSON."""
        self.access_groups = json.dumps(value) if value else None

    @property
    def pricing_override_dict(self) -> dict[str, Any]:
        """Parse pricing override JSON."""
        if not self.pricing_override:
            return {}
        return json.loads(self.pricing_override)

    @pricing_override_dict.setter
    def pricing_override_dict(self, value: dict[str, Any] | None) -> None:
        """Store pricing override as JSON."""
        self.pricing_override = json.dumps(value) if value else None


class Config(Base):
    """Global configuration (LiteLLM destination and sync settings)."""

    __tablename__ = "config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    litellm_base_url: Mapped[str | None] = mapped_column(String, nullable=True)
    litellm_api_key: Mapped[str | None] = mapped_column(String, nullable=True)
    sync_interval_seconds: Mapped[int] = mapped_column(Integer, default=300, nullable=False)
    default_pricing_profile: Mapped[str | None] = mapped_column(String, nullable=True)
    default_pricing_override: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_sync_results: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("sync_interval_seconds >= 0", name="check_sync_interval"),
    )

    @property
    def default_pricing_override_dict(self) -> dict[str, Any]:
        """Parsed global pricing override JSON."""
        if not self.default_pricing_override:
            return {}
        return json.loads(self.default_pricing_override)

    @default_pricing_override_dict.setter
    def default_pricing_override_dict(self, value: dict[str, Any] | None) -> None:
        """Store global pricing override as JSON."""
        self.default_pricing_override = json.dumps(value) if value else None

    @property
    def last_sync_results_dict(self) -> dict[str, Any]:
        """Parsed last sync results JSON."""
        if not self.last_sync_results:
            return {}
        return json.loads(self.last_sync_results)

    @last_sync_results_dict.setter
    def last_sync_results_dict(self, value: dict[str, Any] | None) -> None:
        """Store last sync results as JSON."""
        self.last_sync_results = json.dumps(value) if value else None


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
    system_tags: Mapped[str] = mapped_column(Text, nullable=False, default="[]")  # JSON array
    user_tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    access_groups: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    pricing_profile: Mapped[str | None] = mapped_column(String, nullable=True)
    pricing_override: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON object

    # Ollama-specific
    ollama_mode: Mapped[str | None] = mapped_column(String, nullable=True)

    # Compat model mapping (for compat provider models only)
    mapped_provider_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mapped_model_id: Mapped[str | None] = mapped_column(String, nullable=True)

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
    sync_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

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
    def system_tags_list(self) -> list[str]:
        """Parsed system-generated tags."""
        if not self.system_tags:
            return []
        return json.loads(self.system_tags)

    @system_tags_list.setter
    def system_tags_list(self, value: list[str]) -> None:
        """Store system-generated tags JSON."""
        self.system_tags = json.dumps(value or [])

    @property
    def user_tags_list(self) -> list[str]:
        """Parsed user-defined tags."""
        if not self.user_tags:
            return []
        return json.loads(self.user_tags)

    @user_tags_list.setter
    def user_tags_list(self, value: list[str] | None) -> None:
        """Store user-defined tags JSON."""
        self.user_tags = json.dumps(value) if value else None

    @property
    def access_groups_list(self) -> list[str]:
        """Parse model access_groups JSON to list."""
        if not self.access_groups:
            return []
        return json.loads(self.access_groups)

    @access_groups_list.setter
    def access_groups_list(self, value: list[str] | None) -> None:
        """Store model access_groups as JSON."""
        self.access_groups = json.dumps(value) if value else None

    @property
    def pricing_override_dict(self) -> dict[str, Any]:
        """Parsed pricing override."""
        if not self.pricing_override:
            return {}
        return json.loads(self.pricing_override)

    @pricing_override_dict.setter
    def pricing_override_dict(self, value: dict[str, Any] | None) -> None:
        """Store pricing override as JSON."""
        self.pricing_override = json.dumps(value) if value else None

    @property
    def all_tags(self) -> list[str]:
        """Return merged system and user tags."""
        merged: list[str] = []
        seen = set()
        for tag in (self.system_tags_list or []) + (self.user_tags_list or []):
            if tag not in seen:
                merged.append(tag)
                seen.add(tag)
        return merged

    @property
    def effective_params(self) -> dict[str, Any]:
        """Return LiteLLM-compatible model info merged with user overrides."""
        base_params = dict(self.litellm_params_dict)
        if self.user_params_dict:
            base_params.update(self.user_params_dict)

        if self.all_tags:
            base_params["tags"] = self.all_tags

        return base_params

    def get_effective_access_groups(self) -> list[str]:
        """Return effective access_groups (model overrides provider)."""
        if self.access_groups_list:
            return self.access_groups_list
        if self.provider and self.provider.access_groups_list:
            return self.provider.access_groups_list
        return []

    def get_display_name(self, apply_prefix: bool = True) -> str:
        """Return model name with optional provider prefix."""
        if apply_prefix and self.provider and self.provider.prefix:
            return f"{self.provider.prefix}/{self.model_id}"
        return self.model_id
