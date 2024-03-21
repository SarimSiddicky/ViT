
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import torch
import torchvision.transforms as transforms

# Load the model
model = torch.load('model_architecture.pth',map_location=torch.device('cpu'))
model.load_state_dict(torch.load('model_weights.pth',map_location=torch.device('cpu')))
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

def classify_image():
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        certainty_percentage = torch.max(probabilities, 1)
        certainty_percentage = certainty_percentage[0].item() * 100

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]
        result_label.config(text=f"Predicted class: {class_name}, with Certainty {certainty_percentage}%")

# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=classify_image)
select_button.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(root, text="")
result_label.pack(pady=20)

# Run the main event loop
root.mainloop()
