"""
Application Configuration Management

Uses Pydantic Settings for type-safe configuration with environment variable support.
All settings can be overridden via environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal
from urllib.parse import ParseResult, parse_qs, unquote, urlparse

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

# Project root: src/medical_agent/core/config.py → 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application settings with validation and defaults."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    app_name: str = Field(default="FemTech Medical RAG Agent")
    app_version: str = Field(default="0.1.0")
    environment: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    deployment_mode: Literal["beta", "mvp"] = Field(default="beta")

    # -------------------------------------------------------------------------
    # Server Settings
    # -------------------------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # -------------------------------------------------------------------------
    # Database Settings (PostgreSQL + pgvector)
    # -------------------------------------------------------------------------
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="")
    postgres_db: str = Field(default="phera_postgres")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)

    # Direct database URL (overrides individual postgres settings if set)
    database_url: str | None = Field(default=None)

    @property
    def _parsed_database_url(self) -> ParseResult | None:
        """Parse DATABASE_URL once so callers can safely reuse its parts."""
        if not self.database_url:
            return None
        return urlparse(self.database_url)

    @property
    def _database_query_params(self) -> dict[str, str]:
        """Expose DATABASE_URL query params like Cloud SQL socket host."""
        parsed = self._parsed_database_url
        if not parsed:
            return {}
        return {
            key: values[-1]
            for key, values in parse_qs(parsed.query, keep_blank_values=True).items()
        }

    @property
    def resolved_postgres_user(self) -> str:
        """Database user from DATABASE_URL or POSTGRES_USER fallback."""
        parsed = self._parsed_database_url
        if parsed and parsed.username:
            return unquote(parsed.username)
        return self.postgres_user

    @property
    def resolved_postgres_password(self) -> str:
        """Database password from DATABASE_URL or POSTGRES_PASSWORD fallback."""
        parsed = self._parsed_database_url
        if parsed and parsed.password:
            return unquote(parsed.password)
        return self.postgres_password

    @property
    def resolved_postgres_db(self) -> str:
        """Database name from DATABASE_URL or POSTGRES_DB fallback."""
        parsed = self._parsed_database_url
        if parsed and parsed.path:
            return parsed.path.lstrip("/") or self.postgres_db
        return self.postgres_db

    @property
    def resolved_postgres_host(self) -> str:
        """Database host from DATABASE_URL, including Cloud SQL socket paths."""
        parsed = self._parsed_database_url
        if parsed:
            query_host = self._database_query_params.get("host")
            if query_host:
                return query_host
            if parsed.hostname:
                return parsed.hostname
        return self.postgres_host

    @property
    def resolved_postgres_port(self) -> int:
        """Database port from DATABASE_URL or POSTGRES_PORT fallback."""
        parsed = self._parsed_database_url
        if parsed and parsed.port is not None:
            return parsed.port
        return self.postgres_port

    @property
    def sqlalchemy_async_database_url(self) -> str | URL:
        """Engine-ready async SQLAlchemy URL that preserves socket query params."""
        parsed = self._parsed_database_url
        if not parsed:
            return self.database_connection_string
        return URL.create(
            drivername=parsed.scheme,
            username=self.resolved_postgres_user,
            password=self.resolved_postgres_password,
            host=parsed.hostname,
            port=parsed.port,
            database=self.resolved_postgres_db,
            query=self._database_query_params,
        )

    @property
    def sqlalchemy_sync_database_url(self) -> str | URL:
        """Engine-ready sync SQLAlchemy URL that preserves socket query params."""
        parsed = self._parsed_database_url
        if not parsed:
            return self.sync_database_connection_string
        return URL.create(
            drivername=parsed.scheme.replace("+asyncpg", ""),
            username=self.resolved_postgres_user,
            password=self.resolved_postgres_password,
            host=parsed.hostname,
            port=parsed.port,
            database=self.resolved_postgres_db,
            query=self._database_query_params,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_connection_string(self) -> str:
        """Get the database connection string."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_database_connection_string(self) -> str:
        """Get synchronous database connection string for Alembic."""
        if self.database_url:
            sqlalchemy_url = self.sqlalchemy_sync_database_url
            if isinstance(sqlalchemy_url, URL):
                return sqlalchemy_url.render_as_string(hide_password=False)
            return sqlalchemy_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # -------------------------------------------------------------------------
    # Azure OpenAI Settings - LLM (Chat Completions)
    # -------------------------------------------------------------------------
    azure_openai_api_key: str = Field(default="")
    azure_openai_endpoint: str = Field(default="")
    azure_openai_api_version: str = Field(default="2024-02-15-preview")
    azure_openai_deployment_name: str = Field(default="gpt-4o")
    azure_openai_mini_deployment_name: str = Field(default="gpt-4o-mini")

    # -------------------------------------------------------------------------
    # Azure OpenAI Settings - Embeddings (Separate Endpoint/Key)
    # -------------------------------------------------------------------------
    # Use these if embeddings are deployed in a different Azure OpenAI instance
    azure_openai_embedding_api_key: str = Field(default="")
    azure_openai_embedding_endpoint: str = Field(default="")
    azure_openai_embedding_api_version: str = Field(default="2024-02-15-preview")
    azure_openai_embedding_deployment_name: str = Field(default="text-embedding-3-large")

    # -------------------------------------------------------------------------
    # Docling Parser Settings
    # -------------------------------------------------------------------------
    docling_enable_ocr: bool = Field(
        default=True, description="Enable hybrid OCR for scanned papers"
    )

    # -------------------------------------------------------------------------
    # GCP Settings
    # -------------------------------------------------------------------------
    gcp_project_id: str = Field(default="")
    gcp_bucket_name: str = Field(default="phera_researchpaper")
    gcp_credentials_path: str | None = Field(default=None)
    gcp_cloud_sql_instance: str | None = Field(default=None)

    def is_gcp_configured(self) -> bool:
        """Check if GCP is properly configured."""
        return bool(self.gcp_project_id)

    def is_gcp_storage_configured(self) -> bool:
        """Check if GCP Storage is properly configured."""
        return bool(self.gcp_project_id and self.gcp_bucket_name)

    # -------------------------------------------------------------------------
    # LangSmith Observability Settings
    # -------------------------------------------------------------------------
    langsmith_api_key: str = Field(default="")
    langsmith_tracing: bool = Field(default=False)
    langsmith_project: str = Field(default="phera-agent")

    # -------------------------------------------------------------------------
    # Metadata Extraction Settings
    # -------------------------------------------------------------------------
    metadata_extraction_enabled: bool = Field(
        default=True, description="Enable LLM-based metadata extraction"
    )
    metadata_extraction_model: str = Field(
        default="gpt-4o", description="Model for metadata extraction (GPT-4o for accuracy)"
    )
    metadata_extraction_confidence_threshold: float = Field(
        default=0.7, description="Minimum confidence for extracted metadata"
    )

    # -------------------------------------------------------------------------
    # Chunking Settings
    # -------------------------------------------------------------------------
    chunk_overlap_chars: int = Field(
        default=0, description="No overlap - sections are semantically bounded by Docling"
    )
    respect_section_boundaries: bool = Field(
        default=True, description="Don't split sections mid-concept"
    )

    # -------------------------------------------------------------------------
    # Vector Search Settings
    # -------------------------------------------------------------------------
    # text-embedding-3-large produces 3072-dimensional vectors
    embedding_dimension: int = Field(default=3072)
    vector_similarity_top_k: int = Field(default=10)

    # -------------------------------------------------------------------------
    # Medical Reasoning Settings (pH thresholds)
    # -------------------------------------------------------------------------
    ph_normal_min: float = Field(default=3.8)
    ph_normal_max: float = Field(default=4.5)
    ph_concerning_threshold: float = Field(default=5.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_beta(self) -> bool:
        """Check if running in beta (personalization test) mode."""
        return self.deployment_mode == "beta"

    def is_azure_openai_configured(self) -> bool:
        """Check if Azure OpenAI LLM is properly configured."""
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    def is_azure_openai_embedding_configured(self) -> bool:
        """
        Check if Azure OpenAI embeddings are properly configured.

        Returns True if either:
        1. Separate embedding credentials are provided, OR
        2. Main Azure OpenAI credentials exist (fallback)
        """
        # Use separate embedding credentials if provided
        if self.azure_openai_embedding_api_key and self.azure_openai_embedding_endpoint:
            return True

        # Fall back to main credentials if separate ones not provided
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    def is_langsmith_configured(self) -> bool:
        """Check if LangSmith observability is configured."""
        return bool(self.langsmith_api_key and self.langsmith_tracing)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Clear cache with get_settings.cache_clear() if needed.
    """
    return Settings()


# Global settings instance for convenience
settings = get_settings()
