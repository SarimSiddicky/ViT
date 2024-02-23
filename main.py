
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
from einops import repeat
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from keras.callbacks import EarlyStopping, ReduceLROnPlateau




path_train = r"C:\Users\Sarim&Sahar\OneDrive\Desktop\Science Fair ViT\data\training_data"
path_test = r"C:\Users\Sarim&Sahar\OneDrive\Desktop\Science Fair ViT\data\testing_data"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target

classes = ['healthy', 'mild npdr', 'moderate npdr', 'severe npdr', 'pdr']


for i in classes:
    class_path = os.path.join(path_train, i)
    num_images = len([file for file in os.listdir(class_path) if file.endswith(('jpg', 'jpeg', 'png'))])
    print(f"class: {i}, num of datapoints: {num_images}")



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = []
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if os.path.isdir(cls_path):
                cls_images = [os.path.join(cls_path, img) for img in os.listdir(cls_path)]
                self.images.extend([(img, self.class_to_idx[cls]) for img in cls_images])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

data_transform = transforms.Compose([
    transforms.Resize((144, 144)),

    transforms.ToTensor()
])


dataset = CustomDataset(root_dir=path_train, transform=data_transform)




class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=168):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size **  2) * in_channels, emb_size)
        )

 
        nn.init.xavier_uniform_(self.projection[1].weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


label_mapping = {
    0: "healthy",
    1: "mild npdr",
    2: "moderate npdr",
    3: "severe npdr",
    4: "pdr"
}
def show_images(dataset, num_samples=20, cols=4):
    # Get a random subset of indices
    random_dataset = random.sample(list(range(len(dataset))), num_samples)
    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(random_dataset):
        image, target = dataset[idx]
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(to_pil_image(image[0]))
        plt.colorbar()
        plt.title(label_mapping[target])
        plt.axis('on')

    plt.show()

show_images(dataset)

    
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=0.1)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(q, k, v)
        attn_output = self.norm(self.dropout(attn_output) + x)
        return attn_output

Attention(dim=168, n_heads=12, dropout=0.1)(torch.ones((1, 5,168))).shape



sample_datapoint = torch.unsqueeze(dataset[0][0], 0)
print("Initial shape: ", sample_datapoint.shape)
embedding = PatchEmbedding()(sample_datapoint)
print("Patches shape: ", embedding.shape)



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
norm = PreNorm(168, Attention(dim=168, n_heads=12, dropout=0.1))
norm(torch.ones((1, 5, 168))).shape


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))) + x))
        return x

ff = FeedForward(dim=168, hidden_dim=336)
ff(torch.ones((1, 5, 168))).shape

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

residual_att = ResidualAdd(Attention(dim=168, n_heads=12, dropout=0.1))
residual_att(torch.ones((1, 5, 168))).shape
class ViT(nn.Module):
    def __init__(self, ch=3, img_size=144, patch_size=8, emb_dim=168, n_layers=24, out_dim=5, dropout=0.1, heads=12):
        super(ViT, self).__init__()
        # Attributes
        self.channels = ch
        self.height = img_size  
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        
        # Patching
        self.patch_embedding = PatchEmbedding(in_channels=ch,
                                              patch_size=patch_size,
                                              emb_size=emb_dim)



        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))  
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim)) 


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim)) 

      
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads = heads, dropout = dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout = dropout))))
            self.layers.append(transformer_block)

    
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))


    def forward(self, img):
       
        x = self.patch_embedding(img)
        b, n, _ = x.shape

       
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]


        for i in range(self.n_layers):
            x = self.layers[i](x)

        return self.head(x[:, 0, :])

device = "cpu"
model = ViT().to(device)

model(torch.ones((1, 3, 144, 144)))

train_split = int(0.8 * len(dataset))
train, test = random_split(dataset, [train_split, len(dataset) - train_split])

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32, shuffle=False)
correct_predictions = 0
total_samples = 0


class_weights = []
total_samples = len(dataset)
num_classes = 5

class_counts = [5382, 2443, 5292, 1049, 708]  
total_classes = sum(class_counts)
for count in class_counts:
    class_weight = total_samples / (num_classes * count)
    class_weights.append(class_weight)


class_weights_tensor = torch.tensor(class_weights, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



num_params = count_parameters(model)
print(f"Number of parameters in the model: {num_params}")


earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.2, min_lr=0.00000001)

optimizer = optim.AdamW(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    train_losses = []
    train_correct_predictions = 0
    train_total_samples = 0
    
    model.train()
    for step, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        

        _, predicted = torch.max(outputs, 1)
        train_correct_predictions += torch.sum(predicted == labels).item()
        train_total_samples += labels.size(0)

    train_accuracy = train_correct_predictions / train_total_samples

    model.eval()
    val_losses = []
    val_correct_predictions = 0
    val_total_samples = 0
    

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            val_correct_predictions += torch.sum(predicted == labels).item()
            val_total_samples += labels.size(0)

    val_accuracy = val_correct_predictions / val_total_samples
    
    learning_rate_reduction.step(np.mean(val_losses))
    

    earlystopping(np.mean(val_losses), model)
    if earlystopping.early_stop:
        print("Early stopping")
        break

    print(f">>> Epoch {epoch+1} train loss: {np.mean(train_losses)} train accuracy: {train_accuracy}")
    print(f">>> Epoch {epoch+1} test loss: {np.mean(val_losses)} test accuracy: {val_accuracy}")
    

model.eval()
inputs, labels = next(iter(test_dataloader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)
probabilities = torch.nn.functional.softmax(outputs, dim=1)
_, predicted_classes = torch.max(probabilities, 1)

print(outputs)
print("Predicted classes", predicted_classes)
print("Actual classes", labels)

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

from tkinter import filedialog
from tkinter import *
from PIL import Image

def open_file_dialog():
    root = Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename() 
    root.destroy()  
    return file_path



def preprocess_image(image_path):
    image = Image.open(image_path)
    image = data_transform(image).unsqueeze(0) 
    return image.to(device)

def calculate_accuracy(image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        certainty_percentage, predicted_class = torch.max(probabilities, 1)
        certainty_percentage = certainty_percentage.item() * 100  
        return predicted_class.item(), certainty_percentage, probabilities[0] 


def main():
    image_path = open_file_dialog()
    image_tensor = preprocess_image(image_path)
    predicted_class, certainty_percentage, probabilities = calculate_accuracy(image_tensor)
    print(f"Predicted class: {classes[predicted_class]}")
    print(f"Certainty percentage: {certainty_percentage:.2f}%")
    for i, class_prob in enumerate(probabilities):
        print(f"Certainty percentage for class {classes[i]}: {class_prob.item() * 100:.2f}%")

if __name__ == "__main__":
    main()


