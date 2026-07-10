# Run with: python simple_chatbot/config.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env once here so other chatbot files can import shared settings.
load_dotenv(override=True)

# Read secrets from environment variables instead of hard-coding them.
openai_api_key = os.getenv("OPENAI_API_KEY")
weather_api_key = os.getenv("OPENWEATHER_API_KEY")

if openai_api_key:
    # Print only a short prefix so the full secret never appears in terminal logs.
    print(f"OpenAI API Key found, begins with: {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set in .env")

if weather_api_key:
    print("OpenWeather API Key found")
else:
    print("OpenWeather API Key missing")

# Create a shared OpenAI client for scripts that import this config module.
openai = OpenAI(api_key=openai_api_key)

# Centralize model names so demos can switch models in one place.
CHAT_MODEL = "gpt-4o-mini"
IMAGE_MODEL = "dall-e-3"

# This prompt is reusable for examples that add weather/image tools later.
system_message = (
    "You are WeatherVisionAI, a helpful assistant. "
    "When the user asks about the weather, you must always call get_weather. "
    "The get_weather tool will automatically also generate a city image. "
    "Never answer weather questions without calling get_weather."
)
