import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain_core.prompts import ChatPromptTemplate
import uuid

from complete_pipeline.prompt.query_result import QueryResult


def create_llm(config, max_tokens=None, timeout=None):
    provider = config.get("provider")
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=config["max_retries"],
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=config["max_retries"],
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class LangfusePrompter:
    def __init__(self,
                 system_prompt_name="rag_system_prompt",
                 prompt_name="rag_prompt",
                 max_tokens=None,
                 timeout=None):
        # Initialize Langfuse client (prompt management)
        langfuse = Langfuse()

        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_callback_handler = CallbackHandler()

        # Verify that Langfuse is configured correctly
        assert langfuse.auth_check()
        assert langfuse_callback_handler.auth_check()

        # Retrieve prompts from Langfuse
        langfuse_prompt = langfuse.get_prompt(prompt_name)
        langfuse_system_prompt = langfuse.get_prompt(system_prompt_name)

        # Create LangChain prompt templates from the Langfuse prompts
        self.langchain_prompt_template = ChatPromptTemplate.from_template(
            langfuse_prompt.get_langchain_prompt(),
            metadata={"langfuse_prompt": langfuse_prompt},
        )

        # The system prompt (which might be a long instruction) is kept separate.
        self.system_message = langfuse_system_prompt.prompt

        # Try to create the LLM using the main prompt config; if that fails, use the system prompt config.
        try:
            config = {
                "provider": str(langfuse_prompt.config["provider"]),
                "model": langfuse_prompt.config["model"],
                "temperature": langfuse_prompt.config["temperature"],
                "max_retries": langfuse_prompt.config["max_retries"],
            }
            self.llm = create_llm(config, max_tokens=max_tokens, timeout=timeout)
            print("LLM created using primary prompt config:", self.llm)
        except Exception as e:
            config = {
                "provider": str(langfuse_system_prompt.config.get("provider", "google")),
                "model": langfuse_system_prompt.config["model"],
                "temperature": langfuse_system_prompt.config["temperature"],
                "max_retries": langfuse_system_prompt.config["max_retries"],
            }
            self.llm = create_llm(config, max_tokens=max_tokens, timeout=timeout)
            print("LLM created using system prompt config:", self.llm)

        self.langfuse_callback_handler = langfuse_callback_handler

        # Start conversation history with the system message.
        self.messages = [{"role": "system", "content": self.system_message}]

    # Remove this decorator
    # @observe(as_type="generation")
    def prompt(self, user_input: str, vectordb_context: QueryResult, session_id=None):
        if session_id:
            self.langfuse_callback_handler.session_id = session_id

        formatted_user_message = self.langchain_prompt_template.format(
            user_input=user_input,
            meta_data=vectordb_context.meta_context,
            rag_context=vectordb_context.context,
        )
        self.messages.append({"role": "user", "content": formatted_user_message})

        try:
            langfuse_context.update_current_observation(
                input=self.messages,
                model=self.llm.model,
                metadata=self.llm
            )
        except Exception as e:
            print(f"Langfuse update observation error: {e}")

        response = self.llm.invoke(self.messages, config={"callbacks": [self.langfuse_callback_handler]})

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

    @observe(as_type="generation")
    def _dev_prompt(self, test_input: str, custom_handler):
        """
        Formats the prompt by combining the user input with the retrieved RAG context
        (using the Langfuse prompt template), appends it to the conversation history,
        invokes the LLM, and returns the assistant's response.
        """
        # Append the formatted message to conversation history.
        self.messages.append({"role": "user", "content": test_input})

        # (Optional) update Langfuse observation.
        try:
            langfuse_context.update_current_observation(
                input=self.messages,
                model=self.llm.model,
                metadata=self.llm
            )
        except Exception as e:
            print(f"Langfuse update observation error: {e}")

        # Invoke the LLM with the conversation history.
        response = self.llm.invoke(self.messages, config={"callbacks": [custom_handler]})

        # Update usage details if available.
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

    # Create an instance of Langfuse_Prompter.
    conv = LangfusePrompter()

    # Define the user query and the RAG context (for example, retrieved from a vector DB).
    user_query = "Ayo who are you?"
    dummy_rag_context = QueryResult(
        "no meta",
     "I retrieved this context from the vector database about your capabilities.")

    random_id = uuid.uuid4().hex

    # Get and print the assistant's response.
    print("Assistant Response:\n", conv.prompt(user_query, dummy_rag_context, random_id))
