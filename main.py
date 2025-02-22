from model_training.train_model import train_model
from data_manager.data_manager import DataManager
from predict.predict import predict

if __name__ == "__main__":
    # Train model
    train_model()

    data_manager = DataManager()
    input_data = data_manager.load_data('bike-sharing-dataset.csv')
    input_data = input_data.drop(columns=['cnt'])

    predictions = predict(input_data.loc[[0]])
    print(f'Predictions: {predictions}')