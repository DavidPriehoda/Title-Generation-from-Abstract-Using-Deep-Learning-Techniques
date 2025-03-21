# TitleMaker

A machine learning-based system for generating academic paper titles from abstracts. This was developed as a final year university project.

## Project Overview

TitleMaker uses deep learning models to generate appropriate titles for academic papers based on their abstracts. The system implements and compares two different approaches:

1. **T5 Transformer Model**: A fine-tuned T5 sequence-to-sequence model
2. **BiLSTM Model**: A custom Bidirectional LSTM neural network

The system includes both the model training pipelines and a web-based user interface for generating titles.

- **Backend**: Contains the ML models (T5 and BiLSTM), training scripts, and prediction API
- **Frontend**: A Flask web application that allows users to input paper abstracts and generate titles

**Data Collection**: The dataset is collected using the arXiv API via the script in `Backend/dataset/collect_dataset.py`. This script retrieves paper titles and abstracts from various academic disciplines.

**Data Processing**: The raw data is processed and formatted using `Backend/dataset/create_dataset.py`, which creates a balanced dataset with representation from different academic fields.

## Setup Instructions

### Installation

1. Clone this repository:

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Model Setup

Pretrained models are not included in this repository

**T5 Model Training:**
1. Prepare the dataset:
   ```
   cd Backend/dataset
   python collect_dataset.py  # Downloads data from arXiv API
   python create_dataset.py   # Processes and prepares the training dataset
   ```
   
2. Train the T5 model:
   ```
   cd Backend/T5
   python training.py
   ```
   This will train the T5 model and save it to the `Backend/T5/models/model-t5/` directory.

**BiLSTM Model Training:**
1. Preprocess the data:
   ```
   cd Backend/BiLSTM
   python preprocess_data.py
   ```
   
2. Train the BiLSTM model:
   ```
   python train.py
   ```
   This will train the BiLSTM model and save it to the `Backend/BiLSTM/models/` directory.

### Running the Application

1. Start the backend service:
   ```
   cd Backend/T5
   python server.py
   ```

2. In a new terminal, start the frontend:
   ```
   cd Frontend
   python wsgi.py
   ```

3. Access the web interface at `http://localhost:1337`


## API Documentation

The backend exposes a simple REST API:

### Generate Titles

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "abstract": "The text of the abstract",
  "num_return_sequences": 3,
  "temperature": 25,
  "beam_width": 32
}
```

**Response**:
```json
[
  "Title 1",
  "Title 2",
  "Title 3"
]
```

## Acknowledgements

- Special thanks to the arXiv API for providing access to academic papers
- This project was developed as a final year project at Cardiff University
