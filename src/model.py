# src/model.py
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class ModelTrainer:
    def __init__(self, model=None):
        if model is None:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(random_state=42)
        else:
            self.model = model

    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return the classification report and confusion matrix."""
        y_pred = self.model.predict(X_test)

        # Avoid variable name conflict
        conf_matrix = confusion_matrix(y_test, y_pred)  # Rename the variable
        report = classification_report(y_test, y_pred)
        return report, conf_matrix

    def save_model(self, path):
        """Save the trained model to a file."""
        joblib.dump(self.model, path)
