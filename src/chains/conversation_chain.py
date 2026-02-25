from langchain_classic.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


CONVERSATION_SYSTEM_PROMPT = """You are a helpful assistant.
Use conversation history to keep answers context-aware and coherent.
When context is insufficient, ask one focused clarification question.
Keep answers concise and actionable.
"""


class ConversationQAChain:
    """Multi-turn QA chain with in-memory conversation context."""

    def __init__(self, llm, memory: ConversationBufferMemory | None = None) -> None:
        self.llm = llm
        self.memory = memory or ConversationBufferMemory(
            memory_key="history",
            input_key="input",
            output_key="output",
            return_messages=False,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONVERSATION_SYSTEM_PROMPT),
                (
                    "human",
                    "Conversation history:\n{history}\n\nCurrent user question:\n{question}",
                ),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, question: str) -> str:
        history = self.memory.load_memory_variables({}).get("history", "")
        answer = self.chain.invoke({"history": history, "question": question}).strip()
        self.memory.save_context({"input": question}, {"output": answer})
        return answer

    def clear_memory(self) -> None:
        self.memory.clear()

    def get_history(self) -> str:
        return self.memory.load_memory_variables({}).get("history", "")
