"""Application settings: one validated, cached object. Database URLs are SecretStr, so
repr(), logging and validation errors show '**********' instead of the password."""

from functools import lru_cache
from urllib.parse import urlsplit
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def _host(url:SecretStr) -> str:
    return urlsplit(url.get_secret_value()).hostname or ""

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    # Pooled Neon endpoint (host contains "-pooler"). Used by the running app.
    database_url: SecretStr
    # Direct Neon endpoint (no "-pooler"). Migrations, checkpointer setup(), session features
    database_url_direct: SecretStr

    #Pool sizing
    db_pool_min_size: int = Field(1, ge=0)
    db_pool_max_size: int = Field(5, ge=1)

    # Keep 10s so a Neon cold start can finish. The pool timeout must be longer than the
    # connect timeout, otherwise a cold start is reported as "pool timeout".
    db_connect_timeout_s: int = Field(10, ge=1)
    db_pool_timeout_s : float = Field(20.0, gt=0)

    db_statement_timeout_ms: int = Field(15_000, ge=0)

    # Close idle connections before Neon's 5-minute autosuspend does, and recycle old ones.
    db_max_idle_s: float = Field(240.0, gt=0)
    db_max_lifetime_s: float = Field(1800.0, gt=0)

    @field_validator("database_url", "database_url_direct")
    @classmethod
    def _is_postgres_url(cls, v:SecretStr) -> SecretStr:
        if not v.get_secret_value().startswith(("postgresql://", "postgres://")):
            raise ValueError("must be a postgresql:// connection URL")
        return v

    @model_validator(mode="after")
    def _check_endpoints(self) -> "Settings":
        pooled, direct = _host(self.database_url), _host(self.database_url_direct)
        if pooled.endswith(".neon.tech") and "-pooler" not in pooled:
            raise ValueError("DATABASE_URL must be the pooled Neon host (contains '-pooler')")

        if direct.endswith(".neon.tech") and "-pooler" in direct:
            raise ValueError("DATABASE_URL_DIRECT must be the direct Neon host (no '-pooler')")

        if self.db_pool_min_size > self.db_pool_max_size:
            raise ValueError("DB_POOL_MIN_SIZE cannot exceed DB_POOL_MAX_SIZE")
        if self.db_pool_timeout_s <= self.db_connect_timeout_s:
            raise ValueError("DB_POOL_TIMEOUT_S must be larger than DB_CONNECT_TIMEOUT_S")
        return self

@lru_cache
def get_settings() -> Settings:
    """Build settings once. Inject with Depends(get_settings); in tests use
    app.dependency_overrides or get_settings.cache_clear()."""
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment

    
