from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.tools.document_loader import load_knowledge_base_documents

DEFAULT_COLLECTION_NAME = "smart_qa_knowledge"


def build_vector_store(
    documents,
    embeddings,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    persist_dir: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    persist_directory = str(Path(persist_dir)) if persist_dir else None
    return Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def get_or_create_vector_store(
    knowledge_base_dir: str | Path,
    persist_dir: str | Path,
    embeddings,
    force_rebuild: bool = False,
):
    persist_path = Path(persist_dir)
    collection_name = DEFAULT_COLLECTION_NAME

    if force_rebuild and persist_path.exists():
        shutil.rmtree(persist_path)

    if persist_path.exists() and not force_rebuild:
        try:
            return Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_path),
            )
        except Exception:
            # If loading existing index fails, rebuild from source docs.
            pass

    documents = load_knowledge_base_documents(knowledge_base_dir)
    if not documents:
        raise ValueError(f"No documents found in knowledge base: {knowledge_base_dir}")

    persist_path.mkdir(parents=True, exist_ok=True)
    return build_vector_store(
        documents=documents,
        embeddings=embeddings,
        persist_dir=persist_path,
        collection_name=collection_name,
    )
