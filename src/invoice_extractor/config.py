"""Centralized configuration via Pydantic settings.

A single ``settings`` singleton is loaded from environment variables / ``.env``.
This replaces the old module-global ``config.py`` while preserving the exact
same provider/model resolution behavior.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = three levels up from this file (src/invoice_extractor/config.py).
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings, populated from env vars or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM provider / keys ────────────────────────────────────────────────
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openai_model: str = Field(default="gpt-5.4-mini", alias="OPENAI_MODEL")
    gemini_model: str = Field(default="gemini/gemini-3.5-flash", alias="GEMINI_MODEL")

    max_tokens: int = Field(default=2048, alias="MAX_TOKENS")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")

    # ── OCR ────────────────────────────────────────────────────────────────
    ocr_engine: str = Field(default="paddleocr", alias="OCR_ENGINE")
    paddle_lang: str = Field(default="en", alias="PADDLE_LANG")
    paddle_use_gpu: bool = Field(default=False, alias="PADDLE_USE_GPU")

    # ── Paths / evaluation ─────────────────────────────────────────────────
    dataset_path: Path = Field(
        default=BASE_DIR / "dataset" / "invoices-and-receipts_ocr_v1", alias="DATASET_PATH"
    )
    eval_split: str = Field(default="test", alias="EVAL_SPLIT")
    eval_report_path: Path = Field(
        default=BASE_DIR / "output" / "evaluation_report.json", alias="EVAL_REPORT_PATH"
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def llm_model(self) -> str:
        """Resolve the active model from the selected provider.

        Mirrors the original ``_MODELS`` lookup: gemini -> gemini_model,
        anything else falls back to the openai model.
        """
        if self.llm_provider == "gemini":
            return self.gemini_model
        return self.openai_model

    @computed_field  # type: ignore[prop-decorator]
    @property
    def output_dir(self) -> Path:
        d = BASE_DIR / "output"
        d.mkdir(exist_ok=True)
        return d


settings = Settings()
