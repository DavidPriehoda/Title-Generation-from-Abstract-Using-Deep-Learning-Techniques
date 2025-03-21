import json

from flask import Flask, request, jsonify

from generator import Generator

# Create a Flask app
app = Flask(__name__)


def load_generator(model_path):
    global generator
    generator = Generator(model_path)

# Define a route for the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Extract the input data from the request
    input_data = json.loads(request.get_json())

    # Load the model if it hasn't been loaded yet
    if not generator:
        load_generator(model_path)

    # Generate titles
    titles = generator(input_data['abstract'], input_data['num_return_sequences'], input_data['temperature'], input_data['beam_width'])

    # Return the predictions as JSON
    return jsonify(titles)

# Run the Flask app
if __name__ == '__main__':
    generator = None
    model_path = './models/model-t5/checkpoint-5000'
    app.run(host='0.0.0.0', port=1338)
