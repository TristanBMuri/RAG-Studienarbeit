import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI

class SimpleLLMConversation:
    def __init__(self,
                 model="gemini-2.0-flash-exp",
                 temperature=1,
                 max_tokens=None,
                 timeout=None,
                 max_retries=2):
        # Initialize the LLM wrapper.
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
            "You are not allowed to make up any information."
            "You are a CONCORDIA assistant bot and answer only querries that further that. "
            "Be always polite and friendly"
            "You will receive documents relevant to the query. "
            "If you don't know the answer, just say that you don't know."
            "If you dont get documents that answer the query, just say that you don't know."
        )
        # Start conversation with the system message.
        self.messages = [{"role": "system", "content": self.system_message}]

    def prompt(self, user_input):
        """
        Appends the user message to the conversation history,
        invokes the LLM, appends the response, and returns the result.
        """
        self.messages.append({"role": "user", "content": user_input})
        response = self.llm.invoke(self.messages)
        # Use response.content if available; otherwise, use response directly.
        assistant_response = getattr(response, "content", response)
        self.messages.append({"role": "assistant", "content": assistant_response})
        return assistant_response

# Example usage:
if __name__ == '__main__':
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

    conv = SimpleLLMConversation()
    user_query = "Ayo who are you?"
    print("Assistant Response:\n", conv.prompt(user_query))
