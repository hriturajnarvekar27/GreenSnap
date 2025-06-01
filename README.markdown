# GreenSnap

## Overview

**GreenSnap** is a deep learning project designed to classify images of 15 different vegetables using a pre-trained MobileNetV2 model fine-tuned with PyTorch. The project features a user-friendly web interface built with Streamlit, allowing users to upload vegetable images, view predictions, see nutritional information, and explore confidence scores via an interactive bar chart. The project was developed as a distinct alternative to a similar project, focusing on simplicity, effectiveness, and added features like nutritional data.

### Key Features
- **Classification**: Identifies 15 vegetable classes with high accuracy (99.50% validation accuracy).
- **Nutritional Info**: Displays calories and vitamins for each vegetable.
- **Confidence Scores**: Shows a bar chart of the top 5 prediction probabilities using Plotly.
- **Interactive UI**: Built with Streamlit for easy image uploads and navigation.
- **Modular Code**: Organized into separate files for data loading, model definition, training, and the app.

## Dataset

The dataset consists of 21,000 images across 15 vegetable classes, split into training, validation, and test sets:
- **Classes**: Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato.
- **Total Images**: 21,000 (1,400 per class).
  - **Training**: 14,700 images (70%, 980 per class).
  - **Validation**: 3,150 images (15%, 210 per class).
  - **Test**: 3,150 images (15%, 210 per class).
- **Image Specs**: 224×224 pixels, .jpg format.
- **Obtaining the Dataset**: The dataset is not included in this repository due to its size. You can download it from https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset?resource=download or use a similar vegetable image dataset. Place it in a `dataset/` folder with `train/`, `validation/`, and `test/` subfolders.

## Project Structure

```
GreenSnap/
├── dataset/                    # Not included (see .gitignore)
├── results/                    # Plots and results
│   ├── training_curves.png
│   ├── confusion_matrix.png    # Optional
├── docs/                       # Documentation
│   ├── GreenSnapProjectReport.tex
├── app.py                      # Streamlit app for predictions
├── data_loader.py              # Data loading and preprocessing
├── model.py                    # Model definition (MobileNetV2)
├── train.py                    # Training script
├── confusion_matrix.py         # Optional script for generating a confusion matrix
├── requirements.txt            # Dependencies
├── vegetable_classifier.pth    # Not included (see .gitignore)
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file
```

## Model Performance

The model was trained for 10 epochs using MobileNetV2 (pre-trained on ImageNet) fine-tuned for 15 classes. Training results:
- **Training Accuracy**: 98.87% (Epoch 10).
- **Validation Accuracy**: 99.50% (Epoch 10).
- **Training Loss**: 0.0374 (Epoch 10).
- **Validation Loss**: 0.0193 (Epoch 10).

Training logs:
```
Epoch 1/10, Train Loss: 0.2382, Train Acc: 0.9290, Val Loss: 0.1319, Val Acc: 0.9587
Epoch 2/10, Train Loss: 0.1370, Train Acc: 0.9583, Val Loss: 0.0930, Val Acc: 0.9713
...
Epoch 10/10, Train Loss: 0.0374, Train Acc: 0.9887, Val Loss: 0.0193, Val Acc: 0.9950
```

The model shows excellent generalization, with validation accuracy slightly higher than training accuracy and no signs of overfitting.

## Prerequisites

- **Python**: 3.8–3.10.
- **Hardware**: At least 8GB RAM; GPU recommended for faster training (CUDA-compatible if available).
- **Operating System**: Windows, Linux, or macOS.
- **Dataset**: Place the dataset at `dataset/` with `train/`, `validation/`, and `test/` subfolders.
- **Model Weights**: The `vegetable_classifier.pth` file is not included due to size. You can retrain the model using `train.py` or download the weights from [insert link here, e.g., Google Drive].

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/GreenSnap.git
   cd GreenSnap
   ```

2. **Set Up the Dataset and Model Weights**:
   - Download the dataset and place it in `dataset/` (see "Dataset" section).
   - Download `vegetable_classifier.pth` and place it in the root directory, or retrain the model (see step 4).

3. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```
   - Activate:
     - Windows: `venv\Scripts\activate`
     - Linux/macOS: `source venv/bin/activate`

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - If you have a GPU, ensure PyTorch is installed with CUDA support (see [pytorch.org](https://pytorch.org/get-started/locally/)).
   - For the optional confusion matrix script, install additional dependencies:
     ```bash
     pip install seaborn scikit-learn
     ```

5. **Train the Model** (if `vegetable_classifier.pth` is not available):
   ```bash
   python train.py
   ```
   - This trains the model for 10 epochs, saving `vegetable_classifier.pth` and `results/training_curves.png`.
   - **Expected Time**: ~1–2 hours on a CPU, ~20–30 minutes on a GPU.

6. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   - Access at `http://localhost:8501`.

## Usage

1. **Open the App**:
   - Navigate to `http://localhost:8501` in your browser after running `streamlit run app.py`.

2. **Navigate Pages**:
   - **Home**: Welcome message and project overview.
   - **Prediction**:
     - Upload a vegetable image (.jpg or .png).
     - View the predicted vegetable, confidence score, nutritional info, and a bar chart of the top 5 predictions.
     - Example: Upload a carrot image → Output: “Prediction: Carrot (Confidence: 92.34%)”, nutritional info, and chart.
   - **About**: Project details and technology stack.

3. **Analyze Model Performance** (Optional):
   - Use `confusion_matrix.py` to generate a confusion matrix on the test set:
     ```bash
     python confusion_matrix.py
     ```
   - This creates `results/confusion_matrix.png`, showing where misclassifications occur (if any).

## Deployment (Optional)

To deploy the app online:
1. Push the project to a GitHub repository (already done if following this guide).
2. Sign up at [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub repo and deploy `app.py`.
4. Ensure `vegetable_classifier.pth` is available (e.g., host on Google Drive and download in the app) and `requirements.txt` includes all dependencies.

## Troubleshooting

- **Dataset Path Errors**:
  - Verify the dataset is at `dataset/`.
  - Check that `train/`, `validation/`, and `test/` subfolders contain the 15 class subdirectories (e.g., `train/Carrot/`).
- **Dependency Issues**:
  - If `pip install` fails, try: `pip install --no-cache-dir -r requirements.txt`.
  - For PyTorch GPU support, install the correct version from [pytorch.org](https://pytorch.org/get-started/locally/).
- **Low Accuracy**:
  - If predictions are inaccurate, check for corrupted images in the dataset.
  - Test data loading:
    ```python
    from data_loader import VegetableDataset
    dataset = VegetableDataset("dataset/validation")
    print(dataset.class_names, len(dataset))
    ```
    Expected: 15 class names, ~3,150 images.
- **Streamlit Issues**:
  - Ensure Streamlit is installed (`pip install streamlit`).
  - Run `streamlit run app.py` from the project directory.

## Future Improvements

- Add webcam support for real-time predictions.
- Include more detailed nutritional information or recipes for each vegetable.
- Fine-tune the model further if specific classes are frequently misclassified.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [Streamlit](https://streamlit.io/), and [Plotly](https://plotly.com/).
- Uses MobileNetV2 pre-trained on ImageNet for efficient classification.
- Dataset sourced from a vegetable image collection (update with specific source if known).

---

**Last Updated**: June 2, 2025