import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
from model import get_model
from torchvision import transforms

# Nutritional info (calories and vitamins per 100g)
nutrition_info = {
    "Bean": {"Calories": "31 kcal", "Vitamins": "Vitamin C, K"},
    "Bitter Gourd": {"Calories": "17 kcal", "Vitamins": "Vitamin A, C"},
    "Bottle Gourd": {"Calories": "14 kcal", "Vitamins": "Vitamin C"},
    "Brinjal": {"Calories": "25 kcal", "Vitamins": "Vitamin C, K"},
    "Broccoli": {"Calories": "34 kcal", "Vitamins": "Vitamin C, K"},
    "Cabbage": {"Calories": "25 kcal", "Vitamins": "Vitamin C, K"},
    "Capsicum": {"Calories": "26 kcal", "Vitamins": "Vitamin A, C"},
    "Carrot": {"Calories": "41 kcal", "Vitamins": "Vitamin A, K"},
    "Cauliflower": {"Calories": "25 kcal", "Vitamins": "Vitamin C, K"},
    "Cucumber": {"Calories": "16 kcal", "Vitamins": "Vitamin K"},
    "Papaya": {"Calories": "43 kcal", "Vitamins": "Vitamin A, C"},
    "Potato": {"Calories": "77 kcal", "Vitamins": "Vitamin C, B6"},
    "Pumpkin": {"Calories": "26 kcal", "Vitamins": "Vitamin A, C"},
    "Radish": {"Calories": "16 kcal", "Vitamins": "Vitamin C"},
    "Tomato": {"Calories": "18 kcal", "Vitamins": "Vitamin C, K"}
}

def load_model(num_classes, model_path="vegetable_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def main():
    st.title("Vegetable Classifier")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])
    
    # Class names with underscores to match dataset folder names
    class_names = ["Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", 
                   "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber", 
                   "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"]
    # Display names with spaces for user-friendly output
    display_names = [name.replace("_", " ") for name in class_names]
    model, device = load_model(len(class_names))
    
    if page == "Home":
        st.header("Welcome to the Vegetable Classifier")
        st.write("Upload an image of a vegetable to identify it and learn about its nutritional benefits!")
    
    elif page == "Prediction":
        st.header("Predict a Vegetable")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_tensor = preprocess_image(img_array).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probs).item()
                predicted_class = class_names[predicted_idx]
                display_class = display_names[predicted_idx]
                confidence = probs[predicted_idx].item()
            
            st.write(f"**Prediction**: {display_class} (Confidence: {confidence:.2%})")
            st.write("**Nutritional Info**:")
            st.write(nutrition_info[display_class])
            
            # Confidence score chart
            top_indices = probs.argsort(descending=True)[:5]
            top_probs = probs[top_indices].cpu().numpy()
            top_classes = [display_names[idx] for idx in top_indices]
            fig = px.bar(x=top_probs, y=top_classes, labels={"x": "Confidence", "y": "Vegetable"}, 
                         title="Top 5 Predictions")
            st.plotly_chart(fig)
    
    elif page == "About":
        st.header("About the Project")
        st.write("This project uses a deep learning model to classify vegetables from images.")
        st.write("**Mission**: Assist users in identifying vegetables and learning about their nutritional benefits.")
        st.write("**Technology**: Built with PyTorch, MobileNetV2, and Streamlit for a user-friendly experience.")

if __name__ == "__main__":
    main()