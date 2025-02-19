from langfuse.decorators import observe, langfuse_context
from langchain_google_genai import ChatGoogleGenerativeAI

@observe(as_type="generation")
def gemini_completion(**kwargs):
    """
    This function wraps a call to the Gemini model using ChatGoogleGenerativeAI.
    It logs input parameters and (if available) usage details to Langfuse.
    """
    # Clone kwargs and extract logging information.
    kwargs_clone = kwargs.copy()
    messages = kwargs_clone.pop('messages', None)
    model = kwargs_clone.pop('model', None)
    langfuse_context.update_current_observation(
        input=messages,
        model=model,
        metadata=kwargs_clone
    )

    # Create a ChatGoogleGenerativeAI client with the desired parameters.
    temperature = kwargs.get("temperature", 1)
    max_tokens = kwargs.get("max_tokens", None)
    timeout = kwargs.get("timeout", None)
    max_retries = kwargs.get("max_retries", 2)
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries
    )

    # Invoke the model using the provided messages.
    response = llm.invoke(messages)

    # Update Langfuse observation with usage details if available.
    try:
        langfuse_context.update_current_observation(
            usage_details={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens
            }
        )
    except Exception:
        pass  # Skip if usage details are not available.

    # Return the generated text.
    return response.content if hasattr(response, "content") else response

@observe()
def main():
    """
    Example main function that calls gemini_completion with the Gemini model.
    """
    return gemini_completion(
        model="gemini-2.0-flash-exp",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Gemini!"}
        ]
    )

if __name__ == "__main__":
    result = main()
    print("Gemini API response:\n", result)
