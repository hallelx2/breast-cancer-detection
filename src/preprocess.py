# src/preprocess.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def split_data(self, data, target_column='target', test_size=0.2, random_state=42):
        """Split the dataset into training and test sets."""
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler."""
        # Ensure X_train and X_test are NumPy arrays or DataFrames
        X_train_scaled = self.scaler.fit_transform(X_train.to_numpy())
        X_test_scaled = self.scaler.transform(X_test.to_numpy())
        return X_train_scaled, X_test_scaled
