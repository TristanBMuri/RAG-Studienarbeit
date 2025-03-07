from langfuse import Langfuse

# Initialize the Langfuse client (ensure your environment variables for LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST are set)
langfuse = Langfuse()
langfuse_prompt = langfuse.get_prompt("Test_prompt")

def handle_user_request(user_query):
    # Start a parent trace for the overall RAG request with initial metadata.
    parent_trace = langfuse.trace(
        name="RAG Request",
        input=user_query,
        prompt = "Test_prompt",
        metadata={
            "user_query": user_query,
      }
    )
    parent_trace.generation(prompt=langfuse_prompt)
    try:
        # ----------------------------
        # Document Retrieval Span
        # ----------------------------
        retrieval_span = parent_trace.span(name="Document Retrieval")
        try:
            # Retrieve relevant documents from your vector store or knowledge base.
            documents = retrieve_documents(user_query)
            # Log the number of documents retrieved.
            retrieval_span.update(input="Dummy Input")
            retrieval_span.update(metadata={"docs_retrieved": len(documents)})
            retrieval_span.update(output="Dummy Output")
        except Exception as e:
            # Record any error by adding an "error" key to metadata.
            retrieval_span.update(metadata={"error": f"Retrieval Error: {str(e)}"})
            # Optionally, re-raise or handle the exception.
            documents = []
        # ----------------------------
        # LLM Generation Span
        # ----------------------------
        generation_span = parent_trace.span(name="LLM Generation")
        try:
            # Create a prompt that combines retrieved documents and the user query.
            prompt = create_rag_prompt(documents, user_query)
            # Call the language model with the constructed prompt.
            response = call_llm(prompt)
            generation_span.update(metadata={
                "token_usage": response.token_usage,
                "response_length": len(response.text)
            })
            generation_span.update(
                input="Dummy Input",
                output="Dummy Output"
            )
        except Exception as e:
            generation_span.update(metadata={"error": f"Generation Error: {str(e)}"})
            # Re-raise or handle as needed.
            response = type("Response", (), {"text": "", "token_usage": {}})
        # ----------------------------
        # Update Parent Trace
        # ----------------------------
        parent_trace.update(output= response.text, metadata={"final_response": response.text})
    except Exception as e:
        # If an error occurs at the trace level, update metadata.
        parent_trace.update(metadata={"error": f"Parent Trace Error: {str(e)}"})
    finally:
        # Flush all queued events. (No need to explicitly "finish" spans/traces.)
        langfuse.flush()
    return response.text

# --- Helper functions (replace with your actual implementations) ---

def retrieve_documents(query):
    """
    Retrieve documents related to the user query.
    (This is a placeholder; replace with integration to your document retrieval system.)
    """
    # For example, return a list of document strings.
    return ["Document 1 content", "Document 2 content", "Document 3 content"]

def create_rag_prompt(documents, query):
    """
    Create a prompt that combines the retrieved documents with the user query.
    """
    docs_context = "\n".join(documents)
    prompt = f"Context:\n{docs_context}\n\nQuestion: {query}"
    return prompt

def call_llm(prompt):
    """
    Call the language model with the prompt.
    (This dummy implementation simulates an LLM response.)
    """
    class Response:
        text = "This is the generated answer based on the context and question."
        token_usage = {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
    return Response()

# --- Example usage ---
if __name__ == "__main__":
    user_question = "What is the capital of France?"
    answer = handle_user_request(user_question)
    print("LLM Response:", answer)
