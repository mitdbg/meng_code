import csv
import base64
import requests
from utils import *

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def ask_gpt(image_name):
    answers = None
    
    # Encode the image
    encoded_image = encode_image(image_name)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Based on the plot: 1. Describe the title of the x-axis? 2. Describe the range of the x-axis? 3. Describe the title of the y-axis 4. Describe the range of the y-axis? 5. Describe the different types? Please provide the answers to the aforementioned questions only in the following string format: { \"x-axis\": { \"title\": [INSERT TITLE HERE IN STRING FORMAT], \"range\":  INSERT RANGE HERE IN [start, end] FORMAT ],}, \"y-axis\": { \"title\": [INSERT TITLE HERE IN STRING FORMAT], \"range\":  INSERT RANGE HERE IN [start, end] FORMAT, }, \"types\": [INSERT THE TYPES IN LIST OF LISTS OF LENGTH = 2 WHERE EACH LIST LOOKS LIKE [TYPE NAME, MARKER COLOR in \"black\", \"white\", \"red\",\"purple\",\"green\", \"yellow\", \"blue\", \"pink\", \"orange\", \"grey\"]]}"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Check if the response is successful
    if response.status_code == 200:
        # Parse the response as JSON
        answers = response.json()
    else:
        # Handle errors (non-200 responses)
        print(f"Error: {response.status_code} - {response.text}")

    return answers