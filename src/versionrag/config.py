"""Configuration and logging setup for VersionRAG."""

import logging
import logging.handlers
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "versionrag123"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334

    # OpenAI
    openai_api_key: str = ""

    # Optional separate key for the embedding endpoint. Needed when the main
    # key (e.g. an institute project) has no access to embedding models.
    # Empty -> falls back to openai_api_key.
    openai_embedding_api_key: str = ""

    # OpenRouter — used only when a configured model name contains "/"
    # (e.g. "moonshotai/kimi-k2", "mistralai/mistral-large-2411"). Models
    # without "/" continue to route via direct OpenAI as before. Leave
    # empty to disable; behaviour is identical to the prior single-client
    # setup.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Max concurrent in-flight LLM requests routed via OpenRouter. Direct
    # OpenAI calls are NOT limited by this — only OpenRouter-bound calls
    # (model name contains "/"). Set conservatively to avoid provider
    # rate-limits; bump up if your OpenRouter account allows more.
    openrouter_max_concurrent: int = 10

    # Multi-part document handling for corpora where one logical document
    # (e.g. "Finance Act 2003") is decomposed into many Document nodes
    # (``finance_act_2003_part_1``, ``finance_act_2003_schedule_4_part_2``, …).
    # When True:
    #   - QueryParser expands the LLM-extracted ``document`` to the list of
    #     catalog entries whose name starts with that prefix.
    #   - Vector retriever uses MatchAny across the expanded list.
    #   - Graph retriever unions results across the expanded list.
    # Default False — software corpus uses canonical single-token doc names,
    # so the prefix-expansion code path stays cold and behaviour is bit-
    # identical to the pre-feature build.
    multi_part_documents: bool = False

    # LLM model tiers — two-model setup. gpt-5.4 exclusively as the
    # LLM judge (high-stakes scoring); gpt-5.4-mini everywhere else
    # (parser, generator, indexing-time extraction). Same family, same
    # behaviour profile across the pipeline.
    llm_model_fast: str = "gpt-5.4-mini"
    llm_model_standard: str = "gpt-5.4-mini"
    llm_model_strong: str = "gpt-5.4"
    llm_temperature: float = 0.0

    # LLM model assignments per task
    # All indexing-time extraction (attribute, chat, change) on the mini
    # tier — eval judge is the only place that gets full gpt-5.4.
    llm_model_extract: str = "gpt-5.4-mini"
    llm_model_chat: str = "gpt-5.4-mini"
    llm_model_change: str = "gpt-5.4-mini"

    # Change extraction limits (characters)
    change_max_changelog_chars: int = 6000
    change_max_diff_chars: int = 4000

    # Implicit-change diff chunking: per-LLM-call budget. The DeepDiff is
    # split into chunks at entry boundaries (never inside an entry), then
    # extracted in parallel calls and unioned. Whichever limit is hit
    # first closes the current chunk.
    change_diff_chunk_max_entries: int = 40
    change_diff_chunk_max_chars: int = 8000

    # Semantic diff thresholds for implicit change detection.
    #
    # Tuning history:
    #
    #   v1 (defaults): caps 8/8/5, threshold_unchanged 0.92
    #     → Subtle additions in big transitions were truncated by the cap.
    #       e.g. PYTHON_PRESITE in python_cmdline 3.12→3.13 (41 new chunks)
    #       never reached the LLM.
    #
    #   v2 (current): caps 30/30/15, threshold_unchanged 0.85
    #     → Adaptive cap removed the truncation problem and the new cap
    #       handles subtle edits with cos-sim 0.70-0.85 by routing them
    #       through DeepDiff + the LLM. Captures the bulk of real edits
    #       without flooding the diff extractor with near-identical pairs.
    #
    #   v3 tried (reverted): caps 30/30/15, threshold_unchanged 0.95
    #     → Pushed too many borderline pairs (sim 0.85-0.95) into the diff
    #       extractor. Quality degraded — Change Retrieval (i) dropped
    #       59% → 53% — because the extra pairs added noisy Change nodes
    #       and the right transitions got harder to surface at retrieval.
    #       Reverted to v2 (0.85).
    #
    #   v4 (current): caps 30/30/15, threshold_unchanged 0.92
    #     → Forensic root-cause analysis on the 60 remaining Change(i)
    #       failures showed 70% (42/60) were INDEX_MISSED_SUBTLE_EDIT:
    #       term exists in source file at the GT version but no Change
    #       node exists for the transition (typo fixes, URL changes,
    #       single-line banner removals — chunk-level cos-sim stays >0.85
    #       because most of the chunk is unchanged). v3's 0.95 caught these
    #       but dragged in too much noise; 0.92 sits closer to 0.95 to
    #       catch the bulk of subtle edits while keeping just enough
    #       headroom under 0.95 to avoid the noise that broke v3.
    change_similarity_threshold_new: float = 0.70
    change_similarity_threshold_unchanged: float = 0.92
    change_max_new_chunks: int = 30
    change_max_changed_pairs: int = 30
    change_max_removed_chunks: int = 15

    # Embedding
    embedding_provider: str = "openai"  # "openai" or "local"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-m3"
    local_vector_types: str = "dense,sparse"  # comma-separated: dense, sparse, colbert

    def get_local_vector_types(self) -> set[str]:
        """Parse LOCAL_VECTOR_TYPES into a set. Only relevant for local provider."""
        valid = {"dense", "sparse", "colbert"}
        types = {t.strip().lower() for t in self.local_vector_types.split(",")}
        unknown = types - valid
        if unknown:
            raise ValueError(f"Unknown vector types: {unknown}. Valid: {valid}")
        if "dense" not in types:
            raise ValueError("'dense' must always be included in LOCAL_VECTOR_TYPES")
        return types

    # Chunking
    chunk_min_len: int = 250
    chunk_soft_max: int = 800
    chunk_hard_max: int = 1024
    # "lossless" (default): custom line-based chunker, never drops content,
    #     fence-aware (won't split inside ``` code blocks at soft boundaries).
    # "markdown": original ``markdown_chunker`` library — block-aware
    #     (tables, lists, code) but silently drops 5–60 % of content per
    #     file via aggressive fuzzy dedup + header/footer heuristics.
    chunker: str = "lossless"

    # Qdrant collection name
    qdrant_collection: str = "chunks"

    # Indexing mode — selects how document metadata and change nodes are built.
    #   "full" (default): LLM attribute extraction + LLM change summaries.
    #       Behaviour is bit-identical to the pre-feature build.
    #   "metadata_filtered": deterministic path/filename metadata extraction
    #       and NO change extraction (no Change nodes). This is the
    #       "Metadata-Filtered RAG (Deterministic Version Control)" ablation
    #       baseline — the full VersionRAG architecture with the expensive LLM
    #       summary layer removed. Read once at server startup.
    #   "parser": deterministic metadata extraction AND the LLM change layer.
    #       The cell between the other two: it isolates what the LLM attribute
    #       call actually buys, since the parser derives the same fields from
    #       the path for free. Drops 882 of ~2,200 indexing LLM calls.
    index_mode: str = "full"

    # Logging
    log_level: str = "DEBUG"


def setup_logging(log_level: str = "DEBUG") -> None:
    """Configure application-wide logging with file and console handlers.

    Creates two log files:
    - logs/server.log: DEBUG and INFO messages
    - logs/error.log: ERROR and CRITICAL messages only

    Args:
        log_level: Minimum log level for the console handler (e.g. "DEBUG", "INFO").
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(log_format)

    numeric_level = getattr(logging, log_level.upper(), logging.DEBUG)

    # Root logger captures everything; handlers filter individually
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # --- server.log: DEBUG + INFO ---
    class MaxLevelFilter(logging.Filter):
        """Passes only records at or below a maximum level."""

        def __init__(self, max_level: int) -> None:
            super().__init__()
            self.max_level = max_level

        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno <= self.max_level

    server_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "server.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    server_handler.setLevel(logging.DEBUG)
    server_handler.addFilter(MaxLevelFilter(logging.WARNING))
    server_handler.setFormatter(formatter)

    # --- error.log: ERROR + CRITICAL ---
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "error.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # --- Console: configured level ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(server_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)


settings = Settings()
