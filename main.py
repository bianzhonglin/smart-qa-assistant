import argparse
from pathlib import Path

from src.chains.basic_qa_chain import BasicQAChain
from src.chains.conversation_chain import ConversationQAChain
from src.chains.retrieval_qa_chain import RetrievalQAChain
from src.models.llm_config import LLMConfig
from src.tools.vector_store import get_or_create_vector_store


def run_basic_mode(question: str) -> None:
    config = LLMConfig.from_env()
    llm = config.build_chat_model()
    chain = BasicQAChain(llm)
    answer = chain.invoke(question)
    print("\n[Basic QA Answer]")
    print(answer)


def run_retrieval_mode(question: str, rebuild_index: bool) -> None:
    project_root = Path(__file__).resolve().parent
    knowledge_base_dir = project_root / "data" / "knowledge_base"
    persist_dir = project_root / "outputs" / "vector_store"

    config = LLMConfig.from_env()
    llm = config.build_chat_model()
    embeddings = config.build_embeddings()

    vector_store = get_or_create_vector_store(
        knowledge_base_dir=knowledge_base_dir,
        persist_dir=persist_dir,
        embeddings=embeddings,
        force_rebuild=rebuild_index,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = RetrievalQAChain(llm=llm, retriever=retriever)
    result = chain.invoke(question)

    print("\n[Retrieval QA Answer]")
    print(result["answer"])
    print("\n[Retrieved Sources]")
    for i, doc in enumerate(result["documents"], start=1):
        print(f"{i}. {doc.metadata.get('source', 'unknown')}")


def run_conversation_mode(question: str | None = None) -> None:
    config = LLMConfig.from_env()
    llm = config.build_chat_model()
    chain = ConversationQAChain(llm)

    print("\n[Conversation QA Mode]")
    print("Type your message. Use 'history' to view memory, 'clear' to reset memory, 'exit' to quit.")

    if question:
        answer = chain.invoke(question)
        print(f"\nYou: {question}")
        print(f"Assistant: {answer}")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Conversation ended.")
            break
        if user_input.lower() in {"history", "hist"}:
            history = chain.get_history()
            print("\n[Conversation History]")
            print(history if history else "(No history yet)")
            continue
        if user_input.lower() in {"clear", "reset"}:
            chain.clear_memory()
            print("Conversation memory cleared.")
            continue

        answer = chain.invoke(user_input)
        print(f"Assistant: {answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart QA Assistant")
    parser.add_argument("--mode", choices=["basic", "retrieval", "conversation"], default="basic")
    parser.add_argument("--question", type=str, help="Question text")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild ChromaDB index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "conversation":
        run_conversation_mode(question=args.question)
        return

    question = args.question or input("Enter your question: ").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    if args.mode == "basic":
        run_basic_mode(question)
    else:
        run_retrieval_mode(question, rebuild_index=args.rebuild_index)


if __name__ == "__main__":
    main()
