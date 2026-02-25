from langchain_core.prompts import ChatPromptTemplate


BASIC_QA_SYSTEM_PROMPT = """You are a helpful assistant.
Provide clear answers with short reasoning and practical next steps when helpful.
If the user question is ambiguous, ask one focused clarification question.
"""


def build_basic_qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", BASIC_QA_SYSTEM_PROMPT),
            ("human", "Question: {question}"),
        ]
    )
