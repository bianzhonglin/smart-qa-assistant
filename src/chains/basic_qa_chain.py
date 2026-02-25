from langchain_core.output_parsers import StrOutputParser

from src.prompts.basic_qa_prompt import build_basic_qa_prompt


class BasicQAChain:
    """Basic single-step question answering chain."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = build_basic_qa_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, question: str) -> str:
        return self.chain.invoke({"question": question}).strip()
