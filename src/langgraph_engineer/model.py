import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()

def _get_model(config, default, key):
    model = config['configurable'].get(key, default)
    if model == "openai":
        return ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model == "anthropic":
        return ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620")
    elif model == "gemini":
        google_api_key = os.getenv('GOOGLE_API_KEY')
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,google_api_key=google_api_key)
    else:
        raise ValueError
