from langchain.schema import BaseChatMessageHistory, ChatMessage

class InMemoryChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def clear(self) -> None:
        self.messages = []

# Example usage:
# history = InMemoryChatMessageHistory()
# history.add_message(ChatMessage(role="user", content="Hello!"))
# history.add_message(ChatMessage(role="assistant", content="Hi there!"))
# print(history.get_messages())