"""
Zepto Support Assistant RAG Pipeline (graded baseline)

This module loads Zepto policy documents, creates local embeddings,
stores them in ChromaDB, retrieves relevant policy context, and
routes questions through a LangGraph workflow.

This file is fully deterministic and offline: no API key and no
network call to any LLM provider are required or made here.

The optional real-LLM extension lives in real_llm.py and is only
imported when MOCK_LLM=0.
"""

import os
from pathlib import Path
from typing import TypedDict

import chromadb
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent

DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# MOCK_LLM is the graded baseline.
# Unset or "1" means mock mode (offline, deterministic).
# "0" means the optional real-LLM extension in real_llm.py is used.
MOCK_LLM = os.getenv("MOCK_LLM", "1")
print(f"MOCK_LLM={MOCK_LLM} (1=mock, 0=real-LLM)")

# ---------------------------------------------------------
# Pydantic response model
# ---------------------------------------------------------

class AnswerResponse(BaseModel):
    """
    Defines the required JSON structure returned by the service.

    The answer contains the final response, sources contains the
    retrieved document/chunk IDs, and confidence is always between
    0 and 1.
    """

    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    Stores data passed between LangGraph nodes.

    The state keeps the user's query, intent, retrieved chunks,
    and final validated response while the graph executes.
    """

    query: str
    intent: str
    retrieved_chunks: list[dict]
    response: dict

# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

_embedding_model = None


def get_embedding_model():
    """
    Loads the local SentenceTransformer model only when needed.

    The model used is all-MiniLM-L6-v2 as required by the assignment.
    Loading it once avoids repeatedly creating the model during requests.
    """

    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model


# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------

_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """
    Returns the ChromaDB collection used by the assistant.

    The collection explicitly uses cosine distance so that the
    retrieval step follows the assignment requirement.
    """

    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


# ---------------------------------------------------------
# Document loading and chunking
# ---------------------------------------------------------

def load_documents():
    """
    Loads all eight Zepto policy documents from the docs folder.

    Each document is treated as one chunk because the supplied
    policy documents are short enough for simple per-document
    chunking.
    """

    documents = []

    for file_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        documents.append(
            {
                "id": f"{file_path.stem}_chunk_01",
                "document_id": file_path.stem,
                "text": text,
            }
        )

    return documents


def build_vector_store():
    """
    Embeds the eight policy chunks and stores them in ChromaDB.

    Existing documents are replaced so that repeated application
    starts do not create duplicate chunks in the collection.
    """

    documents = load_documents()

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 policy documents, but found {len(documents)}."
        )

    collection = get_collection()

    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    model = get_embedding_model()

    texts = [item["text"] for item in documents]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[item["id"] for item in documents],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"document_id": item["document_id"]}
            for item in documents
        ],
    )

    return collection


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve_chunks(query: str, top_k: int = 3):
    """
    Retrieves the three most similar policy chunks for a query.

    The query is embedded locally and searched against ChromaDB
    using cosine distance. No LLM or external API is involved.
    """

    collection = get_collection()
    model = get_embedding_model()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        chunks.append(
            {
                "id": chunk_id,
                "document_id": metadata["document_id"],
                "text": document,
                "distance": distance,
            }
        )

    return chunks


# ---------------------------------------------------------
# Node 1 - classify_intent
# ---------------------------------------------------------

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_intent(state: GraphState):
    """
    Classifies the incoming question as policy or general.

    Mock mode (graded baseline) uses the exact required keyword
    heuristic below. When MOCK_LLM=0, this delegates to the optional
    real-LLM classifier in real_llm.py.
    """

    query = state["query"]

    if MOCK_LLM != "0":
        lower_query = query.lower()

        if any(keyword in lower_query for keyword in POLICY_KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"

        return {"intent": intent}

    from real_llm import classify_intent_llm
    return {"intent": classify_intent_llm(query)}

# ---------------------------------------------------------
# Node 2 - retrieve_and_answer
# ---------------------------------------------------------

def retrieve_and_answer(state: GraphState):
    """
    Retrieves relevant Zepto policy chunks and generates an answer.

    Retrieval always uses local embeddings and ChromaDB. Only the
    final answer generation changes between mock and real LLM modes.
    """

    query = state["query"]

    retrieved_chunks = retrieve_chunks(query=query, top_k=3)

    if not retrieved_chunks:
        response = AnswerResponse(
            answer="No relevant Zepto policy information was found.",
            sources=[],
            confidence=0.0,
        )

        return {
            "retrieved_chunks": [],
            "response": response.model_dump(),
        }

    # Required mock branch (graded baseline)
    if MOCK_LLM != "0":
        top_chunk = retrieved_chunks[0]
        snippet = top_chunk["text"][:200]

        answer = f"Based on the retrieved context: {snippet}"

        response = AnswerResponse(
            answer=answer,
            sources=[chunk["id"] for chunk in retrieved_chunks],
            confidence=1.0,
        )

        return {
            "retrieved_chunks": retrieved_chunks,
            "response": response.model_dump(),
        }

    # Optional real-LLM branch
    from real_llm import generate_policy_answer_llm
    response = generate_policy_answer_llm(query, retrieved_chunks)

    return {
        "retrieved_chunks": retrieved_chunks,
        "response": response,
    }


# ---------------------------------------------------------
# Node 3 - direct_answer
# ---------------------------------------------------------

def direct_answer(state: GraphState):
    """
    Handles questions that do not require Zepto policy retrieval.

    Mock mode (graded baseline) returns the required fixed response.
    When MOCK_LLM=0, this delegates to the optional real-LLM path
    in real_llm.py.
    """

    if MOCK_LLM != "0":
        response = AnswerResponse(
            answer=(
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            sources=[],
            confidence=1.0,
        )

        return {"response": response.model_dump()}

    from real_llm import generate_direct_answer_llm
    query = state["query"]

    return {"response": generate_direct_answer_llm(query)}

# ---------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------

def route_intent(state: GraphState):
    """
    Routes policy questions to retrieval and general questions
    to the direct-answer node.
    """

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

def build_graph():
    """
    Builds the required three-node LangGraph workflow.

    The graph first classifies the question, then conditionally
    routes it either to retrieval or to the direct-answer node.
    """

    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


# ---------------------------------------------------------
# Application helper
# ---------------------------------------------------------

def ask_question(query: str):
    """
    Sends a question through the complete LangGraph pipeline.

    The returned dictionary always follows the required
    answer/sources/confidence response structure.
    """

    graph = build_graph()

    result = graph.invoke({"query": query})

    return AnswerResponse.model_validate(result["response"])


# ---------------------------------------------------------
# Initialize ChromaDB
# ---------------------------------------------------------

if __name__ == "__main__":
    build_vector_store()

    print("ChromaDB collection created successfully.")
    print("Stored policy chunks:", get_collection().count())

    print("\nPolicy example:")
    print(
        ask_question("What is the delivery fee?").model_dump_json(indent=2)
    )

    print("\nGeneral example:")
    print(
        ask_question(
            "What is the capital of India?"
        ).model_dump_json(indent=2)
    )
