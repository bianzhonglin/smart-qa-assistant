# LangChain Overview

LangChain is a framework for building applications powered by large language models.
Core building blocks include:
- Models: chat models and embedding models.
- Prompts: reusable templates for stable behavior.
- Chains: workflows that connect prompts, models, and parsers.
- Retrievers: fetch relevant context from external knowledge.
- Memory: track previous interactions in multi-turn conversations.

Common production pattern:
1. Build a reliable base QA flow.
2. Add retrieval over trusted documents.
3. Monitor response quality and retrieval accuracy.
