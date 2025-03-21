import requests
import json

def generate_titles(input_data):
    # Convert the input data to JSON format
    input_data_json = json.dumps(input_data)

    # Define the API endpoint
    url = 'http://127.0.0.1:1338/predict'

    # Send the API request with the input data
    response = requests.post(url, json=input_data_json)

    # Get the response data
    response_data = response.json()

    # Return the predicted titles as a list
    return response_data