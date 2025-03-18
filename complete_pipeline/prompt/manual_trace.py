from langfuse import Langfuse
import os

# get keys for your project from https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-a617199e-4797-4466-8b3b-c14210b3ad21"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-202652e4-08b0-4d2c-af6b-050aff4ec1c4"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

# Assume environment variables for LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY are already set.
langfuse = Langfuse()  # initialize Langfuse client


# Sample knowledge base (list of texts)
documents = [
    "Solar panels absorb sunlight and convert it to electricity during the day.",
    "At night, solar panels do not produce energy; batteries or grid supply power.",
    "Some solar systems use net metering to trade excess daytime power with the grid."
]
# (In a real app, you'd vectorize these docs and store in a vector DB. Here we'll fake a retrieval.)
def answer_question(query: str):
    # Start a new trace for this question
    trace = langfuse.trace(name="RAG Query", input={"question": query}, tags=["RAG"])

    # Span: Document Retrieval
    retrieval = trace.span(name="Document Retrieval", input={"query": query})

    # For demo, "embed" the query (simulated by converting to lowercase words)
    embed_span = retrieval.generation(name="Embed Query", model="mock-embed-model", input={"text": query})
    # (Simulate embedding by splitting the query into words as "vector")
    vector = query.lower().split()
    embed_span.end(output={"vector_preview": vector[:2]}, usage_details={"input": len(query), "output": len(vector), "unit": "chars"})

    # Span: Vector DB Search (simulated by keyword matching)
    search_span = retrieval.span(name="Vector DB Search")
    results = [doc for doc in documents if any(word in doc.lower() for word in vector)]
    search_span.end(output={"documents_found": len(results)})

    retrieval.end(output={"results_count": len(results)})

    # Generation: Answer formulation using an LLM (here we simulate it)
    answer_gen = trace.generation(name="Answer Generation", model="mock-LLM", input={"query": query, "docs": results})
    # (Simulate LLM answer by just combining a couple of docs if available)
    if results:
        answer = " ".join(results[:2])  # simplistic "answer"
    else:
        answer = "I'm not sure. The solar panels might need a battery."
    # End generation span
    answer_gen.end(output={"answer": answer}, usage_details={"input": len(query) + len("".join(results)), "output": len(answer), "unit": "chars"})

    trace.update(output={"answer": answer})
    langfuse.flush()  # ensure trace is sent

    return answer

# Run the example query through the RAG bot
user_q = "Do solar panels work at night?"
response = answer_question(user_q)
print("Bot answer:", response)
