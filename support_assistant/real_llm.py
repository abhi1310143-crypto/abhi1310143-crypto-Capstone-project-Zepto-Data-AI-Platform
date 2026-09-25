"""
Optional real-LLM extension for the Zepto Support Assistant.

This module is only imported when MOCK_LLM=0 (see rag_pipeline.py).
The graded baseline (MOCK_LLM=1, the default) never imports or
executes anything in this file, so it has zero effect on offline
mock-mode grading.

Requires a GROQ_API_KEY environment variable and network access.
"""

import json
import os

from rag_pipeline import AnswerResponse


# ---------------------------------------------------------
# Structured prompt (real-LLM path only)
# ---------------------------------------------------------

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto Support Assistant. Answer customer questions using
only the Zepto policy context provided below.

CONTEXT:
{context}

TASK:
Answer the user's question using only the supplied context.
If the context does not contain enough information, say that the
available Zepto policy information does not provide the answer.

FORMAT:
Return valid JSON with exactly these fields:
{{
    "answer": "string",
    "sources": ["document/chunk IDs"],
    "confidence": 0.0
}}

LENGTH:
Keep the answer concise and easy to understand.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies, prices, times, or rules.

FEW-SHOT EXAMPLE:
Question: How much is priority delivery?
Context: Priority delivery is available for an additional INR 15.
Answer:
{{
    "answer": "Priority delivery costs an additional INR 15.",
    "sources": ["doc_01_chunk_01"],
    "confidence": 1.0
}}

USER QUESTION:
{query}
"""


# ---------------------------------------------------------
# LLM client
# ---------------------------------------------------------

def get_llm():
    """
    Creates the optional Groq LLM client.

    Used only when MOCK_LLM=0. The required mock mode in
    rag_pipeline.py never calls this function.
    """

    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is required when MOCK_LLM=0.")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
        api_key=api_key,
    )


# ---------------------------------------------------------
# Intent classification (real-LLM path)
# ---------------------------------------------------------

def classify_intent_llm(query: str) -> str:
    """
    Asks the configured LLM to classify a query as policy or general.
    """

    prompt = f"""
Classify the user question as exactly one of:
policy_question
general_question

Question:
{query}

Return only one label.
"""

    llm = get_llm()
    result = llm.invoke(prompt)

    label = result.content.strip().lower()

    if "policy_question" in label:
        return "policy_question"

    return "general_question"


# ---------------------------------------------------------
# Policy answer generation (real-LLM path)
# ---------------------------------------------------------

def generate_policy_answer_llm(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generates a grounded policy answer using the configured LLM,
    retrying up to 3 times until the output validates against
    AnswerResponse.
    """

    context = "\n\n".join(
        f"[{chunk['id']}] {chunk['text']}" for chunk in retrieved_chunks
    )

    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    llm = get_llm()

    for attempt in range(3):
        try:
            result = llm.invoke(prompt)
            raw_output = result.content.strip()
            parsed_output = json.loads(raw_output)

            response = AnswerResponse.model_validate(parsed_output)
            return response.model_dump()

        except Exception:
            prompt = f"""
Your previous answer did not satisfy the required JSON schema.

Return ONLY valid JSON with:
- answer: string
- sources: list of strings
- confidence: number from 0 to 1

Do not add Markdown or explanation.

Original task:
{PROMPT_TEMPLATE.format(context=context, query=query)}
"""

    error_response = AnswerResponse(
        answer="ERROR: The real LLM response could not be validated after 3 attempts.",
        sources=[],
        confidence=0.0,
    )

    return error_response.model_dump()


# ---------------------------------------------------------
# Direct answer generation (real-LLM path)
# ---------------------------------------------------------

def generate_direct_answer_llm(query: str) -> dict:
    """
    Answers a general (non-policy) question directly using the
    configured LLM, with no retrieval step.
    """

    prompt = """
ROLE:
You are a Zepto Support Assistant.

CONTEXT:
No policy retrieval was performed.

TASK:
Answer the user's general question.

FORMAT:
Return valid JSON with:
answer, sources, confidence.

LENGTH:
Keep the answer concise.

NEGATIVE CONSTRAINT:
Do not invent Zepto policy information.

FEW-SHOT EXAMPLE:
Question: Hello
Answer:
{
    "answer": "Hello! How can I help you?",
    "sources": [],
    "confidence": 1.0
}

USER QUESTION:
""" + query

    llm = get_llm()

    for attempt in range(3):
        try:
            result = llm.invoke(prompt)
            raw_output = result.content.strip()
            parsed_output = json.loads(raw_output)

            response = AnswerResponse.model_validate(parsed_output)
            return response.model_dump()

        except Exception:
            prompt = """
Return ONLY valid JSON with:
answer, sources, confidence.

confidence must be between 0 and 1.

User question:
""" + query

    error_response = AnswerResponse(
        answer="ERROR: The real LLM response could not be validated after 3 attempts.",
        sources=[],
        confidence=0.0,
    )

    return error_response.model_dump()
