"""Configuration loader and settings dataclasses for Hiver Support Agent."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import yaml
from dotenv import load_dotenv

load_dotenv()

@dataclass
class IngestionConfig:
    raw_path: Path = Path("data/raw/twcs.csv")
    output_dir: Path = Path("data/processed")
    target_brand: str = "AppleSupport"
    split_date: str = "2017-11-01"
    random_seed: int = 42

@dataclass
class RetrievalConfig:
    top_k: int = 5
    max_k: int = 10
    min_relevance_score: float = 0.35
    index_path: Path = Path("data/processed/retrieval_index.pkl")

@dataclass
class EscalationConfig:
    intent_confidence_threshold: float = 0.75
    retrieval_min_score: float = 0.35
    sensitive_keywords: List[str] = field(default_factory=lambda: [
        # Legal, Law Enforcement, & Security / Theft
        "lawsuit", "lawyer", "legal", "stolen", "police", "fraud", "hacked", "security", "compromised", "lost phone", "lost device",
        # Financial & Account Security
        "unauthorized charge", "unauthorized purchase", "billing dispute", "refund", "refunded", "disabled", "locked out", "cancel subscription",
        # Hardware Danger & Physical Repair
        "exploded", "explosion", "smoke", "spark", "swelling", "swollen", "repair", "genius bar appointment", "store appointment",
        # Critical Data Loss
        "lost everything", "lost all my", "lost all of my", "lost all", "deleted all"
    ])

@dataclass
class LLMConfig:
    provider: str = "gemini"  # "gemini" or "mock"
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    random_seed: int = 42
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))

@dataclass
class EvalConfig:
    golden_set_path: Path = Path("data/processed/golden_set.jsonl")
    reports_dir: Path = Path("reports")
    timeout_minutes: int = 15

@dataclass
class AppConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        
        cfg = cls()
        if "ingestion" in raw:
            for k, v in raw["ingestion"].items():
                if hasattr(cfg.ingestion, k):
                    setattr(cfg.ingestion, k, Path(v) if "path" in k or "dir" in k else v)
        if "retrieval" in raw:
            for k, v in raw["retrieval"].items():
                if hasattr(cfg.retrieval, k):
                    setattr(cfg.retrieval, k, Path(v) if "path" in k else v)
        if "escalation" in raw:
            for k, v in raw["escalation"].items():
                if hasattr(cfg.escalation, k):
                    setattr(cfg.escalation, k, v)
        if "llm" in raw:
            for k, v in raw["llm"].items():
                if hasattr(cfg.llm, k):
                    setattr(cfg.llm, k, v)
        if "evaluation" in raw:
            for k, v in raw["evaluation"].items():
                if hasattr(cfg.evaluation, k):
                    setattr(cfg.evaluation, k, Path(v) if "path" in k or "dir" in k else v)
        return cfg

default_config = AppConfig()
