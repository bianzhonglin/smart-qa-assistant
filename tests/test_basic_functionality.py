from pathlib import Path

from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_core.language_models.fake import FakeListLLM

from src.chains.basic_qa_chain import BasicQAChain
from src.chains.conversation_chain import ConversationQAChain
from src.chains.retrieval_qa_chain import RetrievalQAChain
from src.tools.document_loader import load_knowledge_base_documents
from src.tools.vector_store import build_vector_store


def test_knowledge_base_documents_load() -> None:
    project_root = Path(__file__).resolve().parents[1]
    kb_dir = project_root / "data" / "knowledge_base"
    docs = load_knowledge_base_documents(kb_dir)

    assert len(docs) >= 4
    assert all("source" in d.metadata for d in docs)


def test_basic_chain_runs_with_fake_llm() -> None:
    llm = FakeListLLM(responses=["LangChain is used to build LLM-powered applications."])
    chain = BasicQAChain(llm)

    answer = chain.invoke("What is LangChain?")
    assert "LLM-powered" in answer


def test_retrieval_chain_returns_docs_and_answer() -> None:
    project_root = Path(__file__).resolve().parents[1]
    kb_dir = project_root / "data" / "knowledge_base"
    docs = load_knowledge_base_documents(kb_dir)

    embeddings = FakeEmbeddings(size=32)
    vector_store = build_vector_store(docs, embeddings, chunk_size=300, chunk_overlap=50)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    llm = FakeListLLM(responses=["Use retrieval mode when answers must be grounded in your docs."])
    chain = RetrievalQAChain(llm, retriever)

    result = chain.invoke("When should retrieval mode be used?")
    assert "grounded" in result["answer"]
    assert len(result["documents"]) > 0


def test_conversation_chain_maintains_memory() -> None:
    llm = FakeListLLM(responses=["LangChain is a framework.", "You asked what LangChain is."])
    chain = ConversationQAChain(llm)

    first_answer = chain.invoke("What is LangChain?")
    second_answer = chain.invoke("What did I ask before?")

    assert "framework" in first_answer
    assert "asked" in second_answer

    history = chain.get_history()
    assert "Human: What is LangChain?" in history
    assert "AI: LangChain is a framework." in history
    assert "Human: What did I ask before?" in history

    chain.clear_memory()
    assert chain.get_history() == ""
