# Billing FAQ

Q: How can I reduce API cost?
A: Use smaller models, tighter prompts, and retrieval to reduce unnecessary output length.

Q: Why does retrieval increase latency?
A: Retrieval adds embedding lookup and vector search before generation.

Q: Which is cheaper for repetitive internal questions?
A: Retrieval mode is usually more cost-effective because it narrows context to relevant chunks.
