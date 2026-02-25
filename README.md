# Smart QA Assistant
[![CI](https://github.com/bianzhonglin/smart-qa-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/bianzhonglin/smart-qa-assistant/actions/workflows/ci.yml)

A LangChain-based QA assistant supporting three modes:
- Basic QA mode
- Retrieval-augmented QA (RAG) mode
- Conversation QA mode (multi-turn memory)

## Project Structure

```text
smart-qa-assistant/
|- src/
|  |- chains/
|  |  |- basic_qa_chain.py
|  |  |- conversation_chain.py
|  |  `- retrieval_qa_chain.py
|  |- prompts/
|  |  `- basic_qa_prompt.py
|  |- memory/
|  |- tools/
|  |  |- document_loader.py
|  |  `- vector_store.py
|  `- models/
|     `- llm_config.py
|- data/
|  |- knowledge_base/
|  |  |- tech_docs/
|  |  `- product_faq/
|  `- sample_questions/
|- tests/
|  `- test_basic_functionality.py
|- outputs/
|- main.py
|- requirements.txt
|- .env.example
|- architecture_comparison.md
`- README.md
```

## Setup

```bash
cd smart-qa-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then edit `.env`:
- `OPENAI_API_KEY`: required
- `OPENAI_MODEL`: optional, default `gpt-4o-mini`
- `OPENAI_EMBEDDING_MODEL`: optional, default `text-embedding-3-small`
- `OPENAI_TEMPERATURE`: optional, default `0.2`
- `MAX_TOKENS`: optional

## Run

Basic QA:

```bash
python main.py --mode basic --question "What is LangChain used for?"
```

Retrieval QA:

```bash
python main.py --mode retrieval --question "How does vector search help?"
```

Conversation QA (interactive multi-turn):

```bash
python main.py --mode conversation
```

Commands in conversation mode:
- `history`: print full conversation history in memory
- `clear`: clear conversation memory
- `exit`: quit conversation mode

Conversation QA (with first turn provided):

```bash
python main.py --mode conversation --question "Hi, let's talk about LangChain."
```

Rebuild vector index:

```bash
python main.py --mode retrieval --rebuild-index --question "How does vector search help?"
```

## Test

```bash
pytest -q
```

GitHub Actions runs the same test command on pushes and pull requests to `main`.

## Notes
- Retrieval mode stores ChromaDB data under `outputs/vector_store/`.
- Add your own markdown/text docs under `data/knowledge_base/` and rebuild index.
