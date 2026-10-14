from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",         # 忽略.env中多余的变量
        case_sensitive=False,   # OPENAI_API_KEY可匹配openai_api_key
    )

    # ---- 向量库 ----
    vector_store_path: Path = os.getenv("VECTOR_STORE_PATH", PROJECT_ROOT / "data/chroma")
    raw_dir : Path = os.getenv("RAW_DIR", PROJECT_ROOT / "data/raw")

    # ---- Ollama ----
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

    chunk_size: int = 300
    chunk_overlap: int = 30
    top_k: int = 3

settings = Settings()