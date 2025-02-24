from langfuse.decorators import observe, langfuse_context
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Optionally enable Langfuse debug mode via an environment variable.
os.environ["LANGFUSE_DEBUG"] = "true"


@observe(as_type="generation")
def gemini_completion(**kwargs):
    """
    Wraps a call to the Gemini model (via ChatGoogleGenerativeAI) with Langfuse observation.
    Logs input parameters and, if available, usage details.
    """
    kwargs_clone = kwargs.copy()
    messages = kwargs_clone.pop('messages', None)
    model = kwargs_clone.pop('model', None)

    # Update the observation with input and metadata.
    try:
        langfuse_context.update_current_observation(
            input=messages,
            model=model,
            metadata=kwargs_clone
        )
    except Exception as e:
        print(f"Langfuse update observation error: {e}")

    # Create the ChatGoogleGenerativeAI client.
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

    # Invoke the Gemini model with the provided messages.
    response = llm.invoke(messages)

    # Update usage details only if the response includes them.
    if hasattr(response, "usage"):
        try:
            langfuse_context.update_current_observation(
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens
                }
            )
        except Exception as e:
            print(f"Langfuse update usage details error: {e}")
    else:
        print("Response does not have usage attribute; skipping usage update.")

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
