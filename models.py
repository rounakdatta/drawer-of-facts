from dataclasses import dataclass, field
from datetime import datetime
from typing import List

@dataclass
class MetaInformation:
    source: str
    tags: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Information:
    meta: MetaInformation
    info: str
