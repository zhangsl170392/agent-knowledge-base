from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float
    