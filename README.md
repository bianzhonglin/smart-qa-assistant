# Smart QA Assistant

A LangChain-based intelligent Q&A assistant supporting three modes:
- Basic QA mode
- Retrieval-augmented QA (RAG) mode
- Conversation QA mode (multi-turn memory)

## Project Structure

```text
smart-qa-assistant/
├── src/
│   ├── chains/
│   │   ├── basic_qa_chain.py
│   │   ├── conversation_chain.py
│   │   └── retrieval_qa_chain.py
│   ├── prompts/
│   │   └── basic_qa_prompt.py
│   ├── memory/
│   ├── tools/
│   │   ├── document_loader.py
│   │   └── vector_store.py
│   └── models/
│       └── llm_config.py
├── data/
│   ├── knowledge_base/
│   │   ├── tech_docs/
│   │   └── product_faq/
│   └── sample_questions/
├── tests/
│   └── test_basic_functionality.py
├── outputs/
├── main.py
├── requirements.txt
├── .env.example
├── architecture_comparison.md
└── README.md
```

# 克隆或进入项目目录
cd smart-qa-assistant

# 创建Python虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate 
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件，填入你的OpenAI API密钥

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`.

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

Conversation QA (with first turn provided by arg):

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

## Notes
- Retrieval mode stores ChromaDB data under `outputs/vector_store/`.
- Add your own markdown/text docs under `data/knowledge_base/` and rebuild index.
