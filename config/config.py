# config/config.py

import os

class Config:
    """Configuration settings for the application."""
    DATA_PATH = os.getenv('DATA_PATH', './data/')
    MODEL_PATH = os.getenv('MODEL_PATH', './models/')
    LOG_PATH = os.getenv('LOG_PATH', './logs/')
    
    # Model and training settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MODEL_TYPE = 'LinearRegression'
    NESTIMATORS = 100
    
    # Pipeline settings
    PIPELINE_FILE = os.path.join(MODEL_PATH, 'pipeline.pkl')

    # Other settings
    VERBOSE = True
