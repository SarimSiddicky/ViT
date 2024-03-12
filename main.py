import numpy as np
import pandas as pd
import os, random, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, sampler
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import torchmetrics
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torchvision.transforms.functional import to_pil_image
torch.cuda.empty_cache()
path_train = '/kaggle/input/diabetic-retinopathy-resized-arranged'
classes = ['0', '1', '2', '3', '4']


for i in classes:
    class_path = os.path.join(path_train, i)
    num_images = len([file for file in os.listdir(class_path) if file.endswith(('jpg', 'jpeg', 'png'))])
    print(f"class: {i}, num of datapoints: {num_images}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
output_dir = '/kaggle/working/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
   
# set seed
def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    Also, need to add "worker_init_fn=np.random.seed(seed)" in dataloader
    # https://discuss.pytorch.org/t/determinism-in-pytorch-across-multiple-files/156269
    # https://stackoverflow.com/questions/65685060/unique-seed-acrossing-multiple-imported-files-with-random-module-python
    """
    print(f"Setting seeds: {seed} ...... ")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=  True
   
def worker_init_fn(worker_id):    
    '''
    this function is for dataloader's worker_init_fn
    '''                                                    
    np.random.seed(np.random.get_state()[1][0] + worker_id)

set_random_seed(123)

# Define a custom dataset class for loading and preprocessing data
def make_weights_for_balanced_classes(labels):
    count = torch.bincount(torch.tensor(labels)).to(device)
    print('Count:', count.cpu().detach().numpy())
   
    weight = 1. / count.cpu().detach().numpy()
    print('Data sampling weight:', weight)
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)

    return samples_weight



path_val = path_train
path_test = path_train


"""
train_dataloader = DataLoader(path_train, batch_size=128, shuffle=True)
test_dataloader = DataLoader(path_test, batch_size=128, shuffle=False)
"""
# Create data loaders
batch_size = 64
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.5,0.5,0.5]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.5,0.5,0.5]          # Std of ImageNet dataset (used for normalization)


# T.Compose([
#     T.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.2), ratio=(1, 1)),
#     T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.5),
#     T.RandomApply([T.RandomVerticalFlip()], p=0.5),
#     T.RandomRotation(degrees=20),
#     T.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.2), ratio=(1, 1)),
# ])
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
#_____________________
"""
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
])

dataset = datasets.ImageFolder(path_train, transform=transform)

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std


loader = DataLoader(dataset, batch_size=64, shuffle=True)

mean, std = get_mean_std(loader)
print(mean, std)
"""
#_____________________________
train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomAffine(degrees=20, translate=(0, 0), scale=(0.8, 1.2), shear=0),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomVerticalFlip()], p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor(),
    #T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    #T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

dataset = ImageFolder(path_train, transform=train_transform)
dataset_test = ImageFolder(path_train, transform=test_transform)
targets = dataset.targets
                
train_idx, valid_idx= train_test_split(
np.arange(len(targets)),
test_size=0.2,
shuffle=True,
stratify=targets)

train_dataset = Subset(dataset, train_idx)#to_list()
val_dataset = Subset(dataset_test, valid_idx)
test_dataset = Subset(dataset_test, valid_idx)
train_dataset, val_dataset, test_dataset = train_dataset.dataset, val_dataset.dataset, test_dataset.dataset
import matplotlib.pyplot as plt
import random



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
        plt.suptitle(f"Random {num_samples} images from the training dataset", fontsize=16, color="black")

    plt.show()

show_images(dataset)


# For unbalanced dataset we create a weighted sampler                      
weights = make_weights_for_balanced_classes(train_dataset.targets)
weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                        num_workers=2, worker_init_fn=worker_init_fn , sampler=weighted_sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)



# Training loop

best_test_acc = 0
best_epoch = 0
num_classes = 5
train_losses = []  # To store training losses
val_losses = []    # To store validation losses
train_accuracies = []  # To store training accuracies
val_accuracies = []    # To store validation accuracies
logs = ''


# Define the model architecture (ResNet-50 as an example)
print("Model: ViT")
model = models.vit_b_16(weights='IMAGENET1K_V1')

#hidden_layer_size = 512
#num_ftrs = model.fc.in_features
model.heads = nn.Sequential(
   # nn.Linear(num_ftrs, hidden_layer_size),
    #nn.ReLU(),
    # nn.BatchNorm1d(hidden_layer_size),
    #nn.Dropout(0.2),
    nn.Linear(in_features=768, out_features=5, bias=True)
    #nn.Linear(hidden_layer_size, num_classes)
)
#model.load_state_dict(torch.load(r'C:\Users\Sarim&Sahar\OneDrive\Desktop\Science Fair ViT\save_data.pth'), strict=False)
model = model.to(device)

class Temperature(nn.Module):
    def __init__(self, init_weight):
        super(Temperature, self).__init__()
        self.T = nn.Parameter(init_weight)

    def forward(self, x):
        return x / torch.exp(self.T)

# Define loss function and optimizer
count = torch.bincount(torch.tensor(train_dataset.targets)).to(device)
class_weight = len(train_dataset.targets) / count # (count*num_classes)

print('Loss class weight:', class_weight)
class_weight = None
criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
params = list(model.parameters())
optimizer = optim.SGD(params, lr=1e-4, weight_decay=1e-5)#, eps=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, mode='min')

accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='weighted').to(device)
confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true').to(device)
class_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device)
num_epochs = 15

#TRAINING
for epoch in range(num_epochs):
    t = tqdm(enumerate(train_loader, 0), total=len(train_loader),
                smoothing=0.9, position=0, leave=True,
                desc="Train: Epoch: "+str(epoch+1)+"/"+str(num_epochs))
    model.train()
    running_loss = 0.0
   
    for i, (inputs, labels) in t:
        inputs, labels = inputs.to(device).float(),labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels, weight=class_weight)
        criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        outputs = F.softmax(outputs, dim=-1)
        train_accuracy = accuracy(outputs, labels)
   
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
   
    train_accuracy = accuracy.compute()
    train_accuracies.append(float(train_accuracy))
    accuracy.reset()
   

    # Validation
    model.eval()
    val_correct = 0
    val_loss = 0.0
   
    with torch.no_grad():
        t = tqdm(enumerate(val_loader, 0), total=len(val_loader),
                smoothing=0.9, position=0, leave=True,
                desc="Val: Epoch: "+str(epoch+1)+"/"+str(num_epochs))
        for i, (inputs, labels) in t:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, weight=class_weight)
            criterion(outputs, labels)
            val_loss += loss.item()
            outputs = F.softmax(outputs, dim=-1)
            val_accuracy = accuracy(outputs, labels)
            confmat.update(outputs, labels)
            val_class_accuracy = class_accuracy(outputs, labels)
           
   
    val_class_accuracy = class_accuracy.compute()  
    # class_weight = torch.tensor(
    #     [1.0 / ((acc.item() + 1e-5) * cls_count.item()) for acc, cls_count in zip(val_class_accuracy, torch.bincount(torch.tensor(test_dataset.labels)))]
    # ).to(device)  
   
    # Exponential decay for updating class weights based on accuracy
    # accuracy_weights = 0.1 * (1.0 / (val_class_accuracy+1e-5)) + (1 - 0.1) * torch.ones(num_classes).to(device)
    # # Exponential decay for updating class weights based on frequency
    # frequency_weights = 0.1 * (1.0 / count) + (1 - 0.1) * torch.ones(num_classes).to(device)
    # # Combine accuracy and frequency weights
    # class_weight = 0.1 * (accuracy_weights * frequency_weights).to(device) + (1-0.1)*class_weight
    # class_weight = class_weight.to(device)
    # class_weight=None
   
    # print(class_weight)
    # logs+=f'Class loss weight: {list(class_weight.cpu().detach().numpy())}\n'    

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = accuracy.compute()
    val_accuracies.append(float(val_accuracy))

    test_loss = val_loss # test_loss / len(test_loader)
   
    # Calculate metrics for test data
    test_accuracy = val_accuracy
   
    # scheduler
    scheduler.step(val_loss)
    lr_log = f"LR: {optimizer.param_groups[0]['lr']}" # scheduler._last_lr
    print(lr_log)
    logs+=lr_log+'\n'
   
    # Print and log epoch results
    train_results = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Training Accuracy: {train_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
    print(train_results)
    test_results = f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}"
    print(test_results)
    logs+=train_results+'\n'+test_results+'\n'
    # print(confmat.compute())
   
    # Save the model checkpoint
    torch.save(model.state_dict(), os.path.join(output_dir, f'last.pth'))
   
    if (epoch+1)%5==0:
        torch.save(model.state_dict(), os.path.join(output_dir, f'epoch{epoch+1}.pth'))

        # fig, ax = confmat.plot()
        fig, ax = plt.subplots()
        confmat_vals = np.around(confmat.compute().cpu().detach().numpy(), 3)
        im = ax.imshow(confmat_vals)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('True class')

        # Loop over data dimensions and create text annotations.
        for i in range(num_classes):
            for j in range(num_classes):
                text = ax.text(j, i, confmat_vals[i, j],ha="center", va="center", color="black", fontsize=12)

        ax.set_title(f"Confusion Matrix on Test for epoch {epoch+1}")
        fig.savefig(os.path.join(output_dir, f"conf_mat_epoch{epoch+1}.png"))
        plt.close()
       
        # class_weight = 0.3 * (1.0 / (val_class_accuracy+1e-5)) + (1-0.3)*class_weight
        # print(class_weight)
        # logs+=f'Class loss weight: {list(class_weight.cpu().detach().numpy())}\n'  
   
    if best_test_acc <= test_accuracy and epoch!=0:
        best_epoch = epoch+1
        log = f"Improve accuracy from {best_test_acc} to {test_accuracy}"
        print(log)
        logs+=log+"\n"
        best_test_acc = test_accuracy
        torch.save(model.state_dict(), os.path.join(output_dir, f'best.pth'))

        # fig, ax = confmat.plot()
        fig, ax = plt.subplots()
        confmat_vals = np.around(confmat.compute().cpu().detach().numpy(), 3)
        im = ax.imshow(confmat_vals)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('True class')

        # Loop over data dimensions and create text annotations.
        for i in range(num_classes):
            for j in range(num_classes):
                text = ax.text(j, i, confmat_vals[i, j],ha="center", va="center", color="black", fontsize=12)

        ax.set_title("Confusion Matrix on Test for best model")
        fig.savefig(os.path.join(output_dir, "conf_mat_best.png"))
        plt.close()
   
    # resetting all metrics
    accuracy.reset(); class_accuracy.reset(); confmat.reset()
   

# Save the printed outputs to a log.txt file
with open(os.path.join(output_dir, 'log.txt'), 'w') as log_file:
    log_file.write(logs)
    log_file.write(f'Best test accuracy: {best_test_acc} in epoch {best_epoch}')

# Save the loss and accuracy graphs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Validation Accuracy')

plt.savefig(os.path.join(output_dir, 'loss_accuracy_graph.png'))
plt.close()


print("DR classifier model completed")
print("Model saved location :", output_dir)

model.eval()
inputs, labels = next(iter(test_loader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)
print(outputs)
print("Predicted classes", outputs.argmax(-1))
print("Actual classes", labels)
