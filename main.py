import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import openai
import os
import datetime
from data_loader import load_chunkpdf, embed_texts
from Vector_db import QdrantStorage
from Custom_types import RAFSearchResult, RAGChunkAndsrc, RAGUpsertResult, RAGQueryResult

load_dotenv()


inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger= logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="RAG: inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
)

async def rag_inngest_pdf(ctx: inngest.Context):
    def load(ctx: inngest.Context) -> RAGChunkAndsrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_chunkpdf(pdf_path)
        return RAGChunkAndsrc(chunks=chunks, source_id=source_id)

    def _upsert(chunk_and_src: RAGChunkAndsrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunks
        source_id = chunk_and_src.source_id
        vectors = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}: {i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: load(ctx), output_type=RAGChunkAndsrc)
    ingested = await ctx.step.run("upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()



@inngest_client.create_function(
    fn_id="RAG: query pdf",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5):
        query_vector = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vector, top_k)
        return RAFSearchResult(contexts=found["contexts"], sources=found["sources"]) 
    
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAFSearchResult)
    

    context_block = "\n\n".join(f" - {c}"for c in found.contexts)
    user_content =(
        "Use the following context to answer the question. If you don't know the answer, say you don't know.\n\n" \
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using only the above context."
    )

    adapter = openai.AsyncOpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

    async def call_llm():
        response = await adapter.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for answering questions based on provided context."},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1024,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    answer = await ctx.step.run("llm-response", call_llm)
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


app= FastAPI()



inngest.fast_api.serve(app, inngest_client, [rag_inngest_pdf, rag_query_pdf_ai])