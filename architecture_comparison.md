# Architecture Comparison: Basic QA vs Retrieval QA

## Basic QA Chain
- Input: user question
- Flow: prompt -> LLM -> answer
- Strengths: simple setup, low latency, no document pipeline required
- Weaknesses: limited grounding, can hallucinate on domain-specific facts

## Retrieval QA Chain
- Input: user question
- Flow: question -> retriever -> context -> LLM -> answer
- Strengths: grounded answers, source visibility, easy knowledge updates
- Weaknesses: more components, index maintenance, slightly higher latency

## Recommendation
Use basic mode for general conversations and brainstorming.
Use retrieval mode for internal knowledge and factual enterprise Q&A.
