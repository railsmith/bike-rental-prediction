import os
from pipeline.build_pipeline import build_pipeline
from data_manager.data_manager import DataManager
from config.config import Config
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model():
    """Train the machine learning model."""

    if os.path.exists(Config.PIPELINE_FILE):
        print(f"Model already trained and saved at {Config.PIPELINE_FILE}. Skipping training.")
        return

    # Load data
    data_manager = DataManager()
    data = data_manager.load_data('bike-sharing-dataset.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = data_manager.split_data(data)
    
    # Build pipeline and train
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    score = r2_score(y_test, y_pred)
    print("R2 score:", score)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Save the model
    joblib.dump(pipeline, Config.PIPELINE_FILE)
    print(f'Model saved to {Config.PIPELINE_FILE}')
