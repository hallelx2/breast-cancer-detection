from sklearn.datasets import load_breast_cancer
import pandas as pd

class DataLoader:
    @staticmethod
    def load_data():
        """Load the Breast Cancer Wisconsin Dataset"""
        dataset = load_breast_cancer()
        data = pd.DataFrame(dataset.data, columns = dataset.feature_names)
        data['target'] = dataset.target
        return data, dataset.target_names
