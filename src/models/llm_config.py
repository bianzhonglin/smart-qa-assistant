from dataclasses import dataclass
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@dataclass
class LLMConfig:
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.2
    max_tokens: int | None = None
    openai_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_dotenv()
        max_tokens_raw = os.getenv("MAX_TOKENS", "")
        max_tokens = int(max_tokens_raw) if max_tokens_raw.strip() else None
        return cls(
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            max_tokens=max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def require_api_key(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Copy .env.example to .env and set the key.")

    def build_chat_model(self) -> ChatOpenAI:
        self.require_api_key()
        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "openai_api_key": self.openai_api_key,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return ChatOpenAI(**kwargs)

    def build_embeddings(self) -> OpenAIEmbeddings:
        self.require_api_key()
        return OpenAIEmbeddings(model=self.embedding_model, openai_api_key=self.openai_api_key)
