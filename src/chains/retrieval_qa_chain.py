from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


RETRIEVAL_QA_SYSTEM_PROMPT = """You are a retrieval-augmented assistant.
Answer using only the provided context. If context is insufficient, say clearly what is missing.
Keep the answer concise and actionable.
"""


class RetrievalQAChain:
    """Retrieval-augmented QA chain."""

    def __init__(self, llm, retriever) -> None:
        self.llm = llm
        self.retriever = retriever
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RETRIEVAL_QA_SYSTEM_PROMPT),
                (
                    "human",
                    "Question:\n{question}\n\nContext:\n{context}\n\nAnswer based on context:",
                ),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, question: str) -> dict:
        documents = list(self.retriever.invoke(question))
        context = self._build_context(documents)
        answer = self.chain.invoke({"question": question, "context": context}).strip()
        return {"answer": answer, "documents": documents}

    @staticmethod
    def _build_context(documents: list[Document]) -> str:
        if not documents:
            return "No relevant context was retrieved."

        chunks = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            chunks.append(f"[{idx}] source={source}\n{doc.page_content}")
        return "\n\n".join(chunks)
