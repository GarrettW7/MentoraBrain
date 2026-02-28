import os
from functools import lru_cache
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from supabase.client import create_client
from dotenv import load_dotenv

PROMPT_TEMPLATE = """
You are a helpful college Math 1050 professor. 
Answer the student's question based ONLY on the provided context.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
"""


def _find_env_file() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find .env in this directory or any parent directory")


@lru_cache(maxsize=1)
def _build_rag_chain():
    load_dotenv(dotenv_path=_find_env_file())

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not supabase_url or not supabase_service_key or not groq_api_key:
        raise ValueError("Missing SUPABASE_URL, SUPABASE_SERVICE_KEY, or GROQ_API_KEY in .env")

    supabase = create_client(supabase_url, supabase_service_key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=groq_api_key)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def manual_supabase_retriever(query: str) -> str:
        query_embedding = embeddings.embed_query(query)
        rpc_res = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
                "match_count": 5,
            },
        ).execute()

        if not rpc_res.data:
            return "No relevant textbook context found."

        snippets = [
            row.get("content") or row.get("page_content") or ""
            for row in rpc_res.data
        ]
        snippets = [snippet for snippet in snippets if snippet]

        if not snippets:
            return "No relevant textbook context found."

        return "\n\n".join(snippets)

    rag_chain = (
        {
            "context": lambda x: manual_supabase_retriever(x["question"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_math_1050(question: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    rag_chain = _build_rag_chain()
    return rag_chain.invoke({"question": question})


if __name__ == "__main__":
    print("\nAI Professor is thinking (via Direct RPC)...")
    try:
        question = input("Ask a Math 1050 question: ")
        response = ask_math_1050(question)
        print(response)
    except Exception as e:
        print(f"Error: {e}")
