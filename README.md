# Virtualenv Module5 Mini Project

This project demonstrates a machine learning pipeline that processes data, builds a model, and saves it for future use.

## Setup

Follow the instructions below to set up and run the project locally.

### Prerequisites

Make sure you have the following installed:
- Python 3.x
- `pip` (Python's package installer)

### Installation

1. Clone this repository to your local machine.

2. Navigate to the project directory in your terminal.

3. Create a virtual environment:
   ```bash
   python -m venv module5-mini-project

### Install the required dependencies
pip install -r requirements.txt

### Activate the virtual environment:

#### On Windows
    ```bash
    .\module5-mini-project\Scripts\activate
    
#### On macOS/Linux:
    ```bash
    source module5-mini-project/bin/activate

### Running the Project

Once the environment is activated, run the main Python script:

    ```bash
    python .\main.py

    ```code
    R2 score: 0.9880657661164608
    Mean Squared Error: 404.26698914461065
    Model saved to ./models/pipeline.pkl
    Predictions: [138.64]

### Model Saving
The trained model will be saved in the ./models directory as pipeline.pkl for later use.
