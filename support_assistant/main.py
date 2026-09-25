"""
FastAPI application for the Zepto Support Assistant.

The API accepts a user question and sends it through the
LangGraph RAG pipeline. The final response is validated
using the Pydantic AnswerResponse model.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_pipeline import (
    AnswerResponse,
    ask_question,
    build_vector_store,
)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Zepto Support Assistant",
    description="A local RAG-based Zepto policy support service.",
    version="1.0.0",
)


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class AskRequest(BaseModel):
    """
    Defines the input accepted by POST /ask.

    The user provides one query string which is passed to
    the LangGraph support assistant.
    """

    query: str = Field(
        min_length=1,
        description="User's question."
    )


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """
    Builds the local ChromaDB vector store when the API starts.

    The corpus is small, so rebuilding the eight local policy
    embeddings at startup keeps the implementation simple.
    """

    build_vector_store()


# ---------------------------------------------------------
# POST /ask
# ---------------------------------------------------------

@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask(request: AskRequest):
    """
    Answers a user's question using the LangGraph pipeline.

    The returned response contains the answer, source IDs,
    and a confidence value between 0 and 1.
    """

    return ask_question(request.query)
