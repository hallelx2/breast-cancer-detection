# main.py
from src.data_loader import DataLoader
from src.preprocess import Preprocessor
from src.model import ModelTrainer
from src.visualization import Visualization

def main():
    # Load data
    data_loader = DataLoader()
    data, target_names = data_loader.load_data()

    # Preprocess data
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_data(data, target_column='target')
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

    # Train model
    trainer = ModelTrainer()
    trainer.train(X_train_scaled, y_train)
    report, conf_matrix = trainer.evaluate(X_test_scaled, y_test)
    print("Classification Report:", report)

    # Save model
    trainer.save_model('models/best_model.pkl')

    # Visualize results
    Visualization.plot_confusion_matrix(conf_matrix, labels=target_names, output_path='metrics/evaluation_plot.png')

if __name__ == "__main__":
    main()
