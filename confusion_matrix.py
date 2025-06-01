import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import get_data_loaders
from model import get_model

def plot_confusion_matrix(test_dir):
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       _, _, test_loader, class_names = get_data_loaders(
           train_dir=r"D:\College Projects\VegetableClassifier\dataset\train",
           val_dir=r"D:\College Projects\VegetableClassifier\dataset\validation",
           test_dir=test_dir
       )
       
       model = get_model(len(class_names)).to(device)
       model.load_state_dict(torch.load("vegetable_classifier.pth", map_location=device))
       model.eval()
       
       all_preds = []
       all_labels = []
       with torch.no_grad():
           for images, labels in test_loader:
               images, labels = images.to(device), labels.to(device)
               outputs = model(images)
               _, preds = torch.max(outputs, 1)
               all_preds.extend(preds.cpu().numpy())
               all_labels.extend(labels.cpu().numpy())
       
       cm = confusion_matrix(all_labels, all_preds)
       plt.figure(figsize=(10, 8))
       sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
       plt.xlabel("Predicted")
       plt.ylabel("True")
       plt.title("Confusion Matrix")
       plt.xticks(rotation=45, ha="right")
       plt.yticks(rotation=0)
       plt.tight_layout()
       plt.savefig("confusion_matrix.png")
       plt.show()

if __name__ == "__main__":
       test_dir = r"D:\College Projects\VegetableClassifier\dataset\test"
       plot_confusion_matrix(test_dir)