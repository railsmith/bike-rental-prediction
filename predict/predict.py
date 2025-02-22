import os
from config.config import Config
from model_training.train_model import train_model
import joblib

def predict(input_data):
    """Load the trained model and make predictions."""

    if not os.path.exists(Config.PIPELINE_FILE):
        print(f"Trained model not found at {Config.PIPELINE_FILE}. Training the model ...")
        train_model
    
    # Load the model
    model = joblib.load(Config.PIPELINE_FILE)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    return predictions
