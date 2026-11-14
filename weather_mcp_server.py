from mcp.server.fastmcp import FastMCP
import requests
import os
import sys

from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("Weather Server")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

@mcp.tool()
def get_current_weather(city: str):
  """
  Get the current weather for a given city using the OpenWeatherMap API.
  """
  url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
  # print("current weather tool", file=sys.stderr)
  response = requests.get(url)
  if response.status_code == 200:
    data = response.json()
    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"]
    }
  else:
    return {"error": f"Could not retrieve weather data for {city}. Please check the city name and try again."}

@mcp.tool()
def get_weather_forecast(city: str):
  """
  Get the weather forecast for a given city using the OpenWeatherMap API.
  """
  url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
  response = requests.get(url)
  # print("weather forecast tool", file=sys.stderr)
  if response.status_code == 200:
    data = response.json()
    forecast_list = []
    for forecast in data["list"][:5]:  # Get the forecast for the next 5 time intervals (usually 3-hour intervals)
      forecast_list.append({
          "datetime": forecast["dt_txt"],
          "temperature": forecast["main"]["temp"],
          "weather": forecast["weather"][0]["description"],
      })
    return {
      "city": data["city"]["name"],
      "forecast": forecast_list
    }
  else:
    return {"error": f"Could not retrieve weather forecast for {city}. Please check the city name and try again."}


if __name__ == "__main__":
  mcp.run()
