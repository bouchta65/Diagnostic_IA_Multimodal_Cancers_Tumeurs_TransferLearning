# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from ultralytics import YOLO

st.set_page_config(page_title="Diagnostic IA", layout="wide")
st.title("Diagnostic Multimodal par IA")
st.write("Cancers sanguins & Tumeurs cérébrales via Transfer Learning")

# Partie 1 : Classification du cancer sanguin
st.header("Partie 1 : Classification des cancers sanguins")

uploaded_image1 = st.file_uploader("Téléchargez une image de cellule sanguine", type=["png", "jpg", "jpeg"], key="blood")

if uploaded_image1:
    img1 = Image.open(uploaded_image1).convert("RGB")

    model_path = r"B:\Deep Learning\DeepMediScan\data\googlenet_fc_complete.pth"
    model = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img_tensor = test_transform(img1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_class = outputs.argmax(1).item()

    class_names = ['Benign', 'Pre-B', 'Pro-B', 'early Pre-B']
    predicted_class_name = class_names[predicted_class]

    st.success(f"Classe prédite : {predicted_class_name}")

# Partie 2 : Détection de tumeur cérébrale 
st.header("Partie 2 : Détection des tumeurs cérébrales")

uploaded_image2 = st.file_uploader("Téléchargez une image IRM cérébrale", type=["png", "jpg", "jpeg"], key="brain")

if uploaded_image2:
    temp_path = "temp_brain_image.png"
    img2_pil = Image.open(uploaded_image2).convert("RGB")
    img2_pil.save(temp_path)

    yolo_model = YOLO("../models/yolov8_brain_tumor_final.pt")

    results = yolo_model.predict(source=temp_path, conf=0.25, iou=0.45)

    for result in results:
        result_img = result.plot()  
        st.image(result_img, caption="Résultat détection YOLO", use_column_width=True)

        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        class_ids = result.boxes.cls

        st.write("Boxes:", boxes.cpu().numpy())
        st.write("Scores:", scores.cpu().numpy())
        st.write("Class IDs:", class_ids.cpu().numpy())
