"""Application configuration.

Every value is overridable via environment variables (or a .env file) so the same
code runs across dev, CI and prod.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: backend/app/core/config.py -> parents[3]. Local-only training data and
# models live under <repo>/data/ (gitignored), never committed.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # app
    app_name: str = "Accounting AI Assistant"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # database / multi-tenancy
    # Admin role: runs DDL/bootstrap (extension, tables, RLS, roles).
    database_admin_url: str = Field(
        default="postgresql+psycopg://ledgerly:ledgerly@localhost:5432/ledgerly",
        alias="DATABASE_ADMIN_URL",
    )
    # App role: used for all request queries, NOSUPERUSER so RLS is enforced.
    database_url: str = Field(
        default="postgresql+psycopg://ledgerly_app:ledgerly_app@localhost:5432/ledgerly",
        alias="DATABASE_URL",
    )
    app_db_role: str = "ledgerly_app"
    app_db_password: str = Field(default="ledgerly_app", alias="APP_DB_PASSWORD")

    # auth
    jwt_secret: str = Field(
        default="dev-secret-change-me-0000000000000000", alias="JWT_SECRET"
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    # embeddings / vector search — BGE-M3 native dim (shared with the laws RAG)
    embedding_dim: int = 1024

    # LLM (provider-agnostic via LiteLLM)
    # Empty -> the chat uses the deterministic echo stub. Examples:
    #   ollama/llama3.1   (local; set llm_api_base=http://localhost:11434)
    #   claude-3-5-sonnet-latest   (needs ANTHROPIC_API_KEY)
    #   gpt-4o-mini                (needs OPENAI_API_KEY)
    llm_model: str = Field(default="", alias="LLM_MODEL")
    llm_api_base: str = Field(default="", alias="LLM_API_BASE")
    # Provider key (e.g. Gemini/OpenAI/Anthropic). Passed to LiteLLM explicitly so
    # it can live in .env — pydantic loads .env into Settings, not into os.environ.
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_timeout: int = Field(default=60, alias="LLM_TIMEOUT")

    # Fallback model used when no hosted llm_model is configured. In Docker this is the
    # bundled `ollama` Mistral container, so the app has a working LLM out of the box
    # without an API key. Set llm_fallback_enabled=false to force the echo stub instead.
    llm_fallback_enabled: bool = Field(default=True, alias="LLM_FALLBACK_ENABLED")
    llm_fallback_model: str = Field(default="ollama/mistral", alias="LLM_FALLBACK_MODEL")
    llm_fallback_api_base: str = Field(
        default="http://localhost:11434", alias="LLM_FALLBACK_API_BASE"
    )

    # laws RAG (lex) index
    # Build the index on startup if missing, then rebuild on this interval to pick up
    # amendments. Set lex_auto_index=false to manage the index manually.
    lex_auto_index: bool = Field(default=True, alias="LEX_AUTO_INDEX")
    lex_refresh_interval_hours: int = Field(default=168, alias="LEX_REFRESH_INTERVAL_HOURS")

    # OCR
    ocr_languages: str = Field(default="bul+eng", alias="OCR_LANGUAGES")
    ocr_dpi: int = Field(default=300, alias="OCR_DPI")

    # Use a PDF's embedded text layer when it has one (digital PDFs from ERPs, customs,
    # banks). It is exact and fast; OCR is reserved for scanned/image PDFs.
    ocr_prefer_embedded_text: bool = Field(default=True, alias="OCR_PREFER_EMBEDDED_TEXT")

    # OCR image preprocessing (only applied when OpenCV is installed).
    ocr_preprocess: bool = Field(default=True, alias="OCR_PREPROCESS")
    ocr_deskew: bool = Field(default=True, alias="OCR_DESKEW")
    ocr_denoise: bool = Field(default=True, alias="OCR_DENOISE")
    ocr_threshold: str = Field(default="otsu", alias="OCR_THRESHOLD")  # otsu | adaptive | none
    # Words OCR'd below this confidence (0..1) are flagged so extraction can
    # down-weight them and try a register/vision recovery.
    ocr_word_conf_min: float = Field(default=0.60, alias="OCR_WORD_CONF_MIN")

    # LLM vision fallback for poorly scanned pages / photos (needs a vision-capable model).
    ocr_vision_fallback: bool = Field(default=True, alias="OCR_VISION_FALLBACK")
    ocr_vision_model: str = Field(default="", alias="OCR_VISION_MODEL")  # "" -> reuse llm_model
    # Where the vision model is served; lets a self-hosted vision model (e.g. ollama) be
    # used independently of the text LLM. "" -> reuse llm_api_base.
    ocr_vision_api_base: str = Field(default="", alias="OCR_VISION_API_BASE")
    # Page mean confidence below which the vision model is consulted.
    ocr_vision_conf_min: float = Field(default=0.75, alias="OCR_VISION_CONF_MIN")

    # Commercial-register lookup by EIK (scrapes web.company.guru). Cached, times out
    # gracefully and no-ops when disabled. Enabling it sends EIKs to a third party.
    company_lookup_enabled: bool = Field(default=True, alias="COMPANY_LOOKUP_ENABLED")
    company_lookup_base_url: str = Field(
        default="https://web.company.guru", alias="COMPANY_LOOKUP_BASE_URL"
    )
    company_lookup_timeout: float = Field(default=5.0, alias="COMPANY_LOOKUP_TIMEOUT")
    company_lookup_cache_size: int = Field(default=512, alias="COMPANY_LOOKUP_CACHE_SIZE")

    # Trainable preprocessing (assist-only; amounts stay deterministic). The labeled
    # dataset, OCR-text cache and trained model live under a gitignored local path.
    preprocessing_data_dir: str = Field(
        default=str(_REPO_ROOT / "data" / "preprocessing"), alias="PREPROCESSING_DATA_DIR"
    )
    # Doc-type classifier: used only when the keyword detector is unsure and the model is
    # confident; decisive keyword phrases always win.
    doctype_classifier_enabled: bool = Field(default=True, alias="DOCTYPE_CLASSIFIER_ENABLED")
    doctype_model_path: str = Field(
        default=str(_REPO_ROOT / "data" / "models" / "doctype.joblib"), alias="DOCTYPE_MODEL_PATH"
    )
    doctype_model_min_proba: float = Field(default=0.65, alias="DOCTYPE_MODEL_MIN_PROBA")
    # LLM few-shot assist for weak NON-AMOUNT fields (type/parties); never touches money.
    llm_assist_enabled: bool = Field(default=True, alias="LLM_ASSIST_ENABLED")
    llm_assist_examples: int = Field(default=4, alias="LLM_ASSIST_EXAMPLES")
    # Opt-in: a user "save for training" writes the corrected document into the local
    # dataset. Off by default since it copies tenant documents to local disk.
    training_capture_enabled: bool = Field(default=False, alias="TRAINING_CAPTURE_ENABLED")
    # Learned field selectors (amounts/parties/number/date/direction). They only PICK
    # among deterministic candidates — values stay verbatim — and override a field only
    # when confident and the deterministic result was weak. Gitignored, local artifacts.
    field_models_enabled: bool = Field(default=True, alias="FIELD_MODELS_ENABLED")
    field_models_dir: str = Field(
        default=str(_REPO_ROOT / "data" / "models" / "fields"), alias="FIELD_MODELS_DIR"
    )
    field_model_min_proba: float = Field(default=0.6, alias="FIELD_MODEL_MIN_PROBA")

    # comparison / IR (TF-IDF fusion)
    ir_weight_word: float = Field(default=0.65, alias="IR_WEIGHT_WORD")
    ir_weight_char: float = Field(default=0.35, alias="IR_WEIGHT_CHAR")
    ir_word_ngram: tuple[int, int] = (1, 2)
    ir_char_ngram: tuple[int, int] = (3, 5)
    # Score at/above which two documents are flagged a likely duplicate.
    ir_duplicate_threshold: float = Field(default=0.85, alias="IR_DUPLICATE_THRESHOLD")
    # Score below which a "best match" is treated as no match.
    ir_match_threshold: float = Field(default=0.10, alias="IR_MATCH_THRESHOLD")

    # validation
    # Absolute currency tolerance for arithmetic checks (rounding slack).
    validation_amount_tolerance: float = Field(
        default=0.02, alias="VALIDATION_AMOUNT_TOLERANCE"
    )
    # Plausible Bulgarian VAT rates: 0% (exempt/reverse-charge), 9% (reduced), 20% (standard).
    validation_valid_vat_rates: list[float] = Field(
        default_factory=lambda: [0.0, 0.09, 0.20],
        alias="VALIDATION_VALID_VAT_RATES",
    )

    @property
    def ir_weights_normalized(self) -> tuple[float, float]:
        """Word/char weights normalized to sum to 1.0."""
        total = self.ir_weight_word + self.ir_weight_char
        if total <= 0:
            return 0.5, 0.5
        return self.ir_weight_word / total, self.ir_weight_char / total


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (one instance per process)."""
    return Settings()
