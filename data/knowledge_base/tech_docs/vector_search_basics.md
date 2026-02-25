# Vector Search Basics

Vector search converts text into embeddings and retrieves semantically similar chunks.

Recommended setup:
- Split long documents into chunks with overlap.
- Store chunks in a vector database (ChromaDB in this project).
- Retrieve top-k chunks for each question.
- Send question + retrieved context to the LLM.

Benefits:
- Better factual grounding.
- Lower hallucination risk.
- Faster updates by changing documents without retraining model.
