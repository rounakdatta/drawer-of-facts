from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class MetaInformation:
    source: str
    timestamp: datetime
    tags: List[str]

@dataclass
class Information:
    meta: MetaInformation
    info: str
