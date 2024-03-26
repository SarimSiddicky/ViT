import gradio as gr
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import torch
import torchvision.transforms as transforms

# Load the model
model = torch.load('model_architecture (4).pth',map_location=torch.device('cpu'))
model.load_state_dict(torch.load('model_weights (4).pth',map_location=torch.device('cpu')))
model.eval()

# Define the classes (replace with your actual class names)
classes = ['healthy', 'mild npdr', 'moderate npdr', 'severe npdr', 'pdr']

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model.eval()

def classify_image(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return {classes[i]: probabilities[0][i].item() for i in range(len(classes))}

# Launch the Gradio interface
gr.Interface(fn=classify_image, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=5), title="Diabetic Retinopathy Classifier").launch()
