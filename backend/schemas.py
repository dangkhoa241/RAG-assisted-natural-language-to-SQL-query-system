"""Request bodies. Responses are plain dicts; their shape is documented in the README ("Running the app")."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUESTION_CHARS = 500


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    mode: Literal["auto", "llm", "rule_based"] = None    # None = the server's DEFAULT_MODE
    use_glossary: bool = None                            # None = the server's GLOSSARY_DEFAULT

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("question is empty")
        return v
