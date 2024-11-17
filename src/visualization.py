# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    @staticmethod
    def plot_confusion_matrix(conf_matrix, labels, output_path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(output_path)
        plt.close()
