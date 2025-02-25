import getpass
import os

import langfuse
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.decorators import observe, langfuse_context

from langfuse.callback import CallbackHandler
from langfuse import Langfuse

class SimpleLLMConversation:
    def __init__(self,
                 model="gemini-2.0-flash-exp",
                 temperature=1,
                 max_tokens=None,
                 timeout=None,
                 max_retries=2):
        # Langfuse
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        self.langfuse_callback_handler = CallbackHandler()

        # Initialize the LLM wrapper using ChatGoogleGenerativeAI.
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        # Define the initial system message.
        self.system_message = (
            "You are an expert assistant specialized in returning truthful responses to user questions. "
            "You are not allowed to make up any information. "
            "You are a CONCORDIA assistant bot and answer only queries that further that. "
            "Be always polite and friendly. "
            "You will receive documents relevant to the query. "
            "If you don't know the answer, just say that you don't know. "
            "If you don't get documents that answer the query, just say that you don't know. "
        )
        # Start conversation with the system message.
        self.messages = [{"role": "system", "content": self.system_message}]

    @observe(as_type="generation")
    def prompt(self, user_input):
        """
        Appends the user message to the conversation history,
        invokes the LLM, appends the response, and returns the result.
        """
        self.messages.append({"role": "user", "content": user_input})

        # Update Langfuse observation with usage details, if available.
        try:
            langfuse_context.update_current_observation(
                input=self.messages,
                model=self.llm.model,
                metadata=self.llm
            )
        except Exception as e:
            print(f"Langfuse update observation error: {e}")
        response = self.llm.invoke(self.messages, config={"callbacks": [self.langfuse_callback_handler]})

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

        assistant_response = getattr(response, "content", response)
        self.messages.append({"role": "assistant", "content": assistant_response})


        return assistant_response


# Example usage:
if __name__ == '__main__':
    # If the GOOGLE_API_KEY is not set, prompt for it.
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

    conv = SimpleLLMConversation()

    user_query = "Ayo who are you?"
    print("Assistant Response:\n", conv.prompt(user_query))
