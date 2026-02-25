from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents_from_folder(folder_path: str | Path) -> list[Document]:
    folder = Path(folder_path)
    if not folder.exists():
        return []

    documents: list[Document] = []
    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path.as_posix()),
                    "filename": file_path.name,
                    "category": folder.name,
                },
            )
        )
    return documents


def load_knowledge_base_documents(base_dir: str | Path) -> list[Document]:
    base = Path(base_dir)
    return load_documents_from_folder(base / "tech_docs") + load_documents_from_folder(base / "product_faq")
