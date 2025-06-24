import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn


model = models.vit_b_16(weights=None)
model.heads = nn.Sequential(nn.Linear(768, 5)) 

weights_path = "best.pth"
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

classes = ['healthy', 'mild npdr', 'moderate npdr', 'severe npdr', 'pdr']

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_image(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return {classes[i]: probabilities[0][i].item() for i in range(len(classes))}

# Main
gr.Interface(fn=classify_image, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=5), title="Diabetic Retinopathy Classifier").launch()
