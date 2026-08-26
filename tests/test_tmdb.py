import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("tmdb_api")


if not api_key:
    print("Error: tmdb_api not found in .env")
    exit(1)

url = f"https://api.themoviedb.org/3/movie/500?api_key={api_key}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(f"API key works! Test query returned: {data['title']} ({data['release_date']})")
else:
    print(f"API key failed. Status code: {response.status_code}")
    print(response.json().get("status_message", "Unknown error"))