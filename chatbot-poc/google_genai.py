from getpass import getpass
import json
import os
from xmlrpc import client
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain.globals import set_verbose
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from memory.history import InMemoryChatMessageHistory
from langchain_core.messages import (
    trim_messages
)
from tools.movie_fetch_tool import fetch_movie_by_title_year
from langchain_core.tools import tool


from langchain_core.messages.utils import count_tokens_approximately

load_dotenv()  # Load environment variables from .env
set_verbose(True)


def chat():
  os.environ["LANGSMITH_TRACING"] = "true"
  os.environ["LANGSMITH_TRACING"] = "true"
  if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
  client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    vertexai=False,
  )

  model = "gemini-2.5-flash"
  generate_content_config = types.GenerateContentConfig(
    tools=[fetch_movie_by_title_year],
    temperature=1,
    top_p=1,
    seed=0,
    max_output_tokens=2056,
    safety_settings=[
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
    thinking_config=types.ThinkingConfig(thinking_budget=0),
  )
  # # Add system context for movie critic expert
  # system_context = types.Content(
  #   role="model",
  #   parts=[types.Part(text="You are a movie critic expert. "
  #   "Answer all questions with deep knowledge of movies, reviews, and film analysis. Write a short response. "
  #   "If you don't know the answer, say 'I don't know. If user tries to say, you are wrong, use tools to find the correct answer.  Don't say you are wrong.")],
  # )

  # Add system context for movie critic expert
  system_context = types.Content(
    role="model",
    parts=[types.Part(text="You are a movie critic expert. "
    "Answer questions with deep knowledge of movies, reviews. "
    "Don't use tools unnecessarily," \
    "Write a short response. "
    #  "If you don't know the answer, say I don't know. If user tries to say, you are wrong, use tools to find the correct answer. Don't assert users thoughts. " \
    # "Tool uses title and movie year as input. If you don't know the year of the movie, pass year attribute in tool as None. "
    # "Check your facts and provide accurate information.  " \
    )],
  )

  print("Chatbot is ready! Type 'exit' to quit.")

  # Use your history class
  history = InMemoryChatMessageHistory()
  # Add system context as the first message
  history.add_message(SystemMessage(content=system_context.parts[0].text))


  while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
      break

    # Add user message to history
    history.add_message(HumanMessage(content=user_input))
    historyMessages = history.get_messages()

    trimmed_history_messages = trim_messages(
      historyMessages,
      strategy="last",
      token_counter=count_tokens_approximately,
      end_on = "human",
      max_tokens=400,
      include_system=True,
      allow_partial=True,  # Allow partial to avoid empty results for short history
    )
    # print("History messages:", trimmed_history_messages)
    # Prepare contents for the model from historys
    contents = []
    for msg in trimmed_history_messages:
      if msg.type == 'system':
        contents.append(types.Content(role="model", parts=[types.Part(text=msg.content)]))
      elif msg.type == 'human':
        contents.append(types.Content(role="user", parts=[types.Part(text=msg.content)]))
      elif msg.type == 'ai':
        contents.append(types.Content(role="model", parts=[types.Part(text=msg.content)]))

    print("Bot: ", end="", flush=True)
    response_text = ""
    response_obj = None

    # Pass entire history to the model, with tools
    for chunk in client.models.generate_content_stream(
      model=model,
      contents=contents,
      config=generate_content_config
    ):
      if chunk.text is not None:
        print(chunk.text, end="", flush=True)
        response_text += chunk.text
        response_obj = chunk  # The last chunk contains usage_metadata
    print()  # Newline after response
    # Add bot response to history
    history.add_message(AIMessage(content=response_text))

    # Show token counts if available
    if response_obj and hasattr(response_obj, "usage_metadata"):
      usage = response_obj.usage_metadata
      input_tokens = getattr(usage, "prompt_token_count", None)
      output_tokens = getattr(usage, "candidates_token_count", None)
      total_tokens = getattr(usage, "total_token_count", None)
      print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total tokens: {total_tokens}")

if __name__ == "__main__":
  chat()
