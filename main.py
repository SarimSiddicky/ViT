import numpy as np
import cv2
import numpy as np
from PIL import Image
import os, random, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import torchmetrics
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split



torch.cuda.empty_cache()
path_train = '/kaggle/input/diabetic-retinopathy-resized-arranged'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
output_dir = '/kaggle/working/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def set_random_seed(seed: int) -> None:
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

    np.random.seed(np.random.get_state()[1][0] + worker_id)

set_random_seed(123)

def make_weights_for_balanced_classes(labels):
    count = torch.bincount(labels).to(device)
    print('Count:', count.cpu().detach().numpy())

    weight = 1. / count.cpu().detach().numpy()
    print('Data sampling weight:', weight)
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)

    return samples_weight


path_test = path_train



batch_size = 64
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406] 
IMAGENET_STD = [0.229, 0.224, 0.225]         

        
train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

all_dataset = ImageFolder(path_train, transform=train_transform)
dataset_test = ImageFolder(path_train, transform=test_transform)

targets = all_dataset.targets

train_idx, test_idx = train_test_split(
np.arange(len(targets)),
test_size=0.2,
shuffle=True,
stratify=targets)
print(len(train_idx), len(test_idx))

train_dataset = Subset(all_dataset, train_idx)
val_dataset = Subset(dataset_test, test_idx)
test_dataset = Subset(dataset_test, test_idx)

labels =  torch.tensor(all_dataset.targets)[train_dataset.indices]
count = torch.bincount(labels).to(device)
print('Train Count:', count.cpu().detach().numpy())

labels = torch.tensor(all_dataset.targets)[val_dataset.indices]
count = torch.bincount(labels).to(device)
print('Validation Count:', count.cpu().detach().numpy())

labels = torch.tensor(all_dataset.targets)[test_dataset.indices]
count = torch.bincount(labels).to(device)
print('Test Count:', count.cpu().detach().numpy())
# sys.exit()
import torchvision

label_mapping = {
    0: "healthy",
    1: "mild npdr",
    2: "moderate npdr",
    3: "severe npdr",
    4: "pdr"
}
def inverse_normalize(tensor, mean, std):
    '''
    does not work for batch of images, only works on single image 
    tensor shape: (1, nc, h, w)
    '''
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def imshow(img):
    img = img / 2 + 0.5
    inverse_normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
images, labels = next(iter(train_loader))

imshow(torchvision.utils.make_grid(images))


weights = make_weights_for_balanced_classes(torch.tensor(all_dataset.targets)[train_dataset.indices])
weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                        num_workers=4, worker_init_fn=worker_init_fn , sampler=weighted_sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)


best_test_acc = 0
best_epoch = 0.00
.000000
num_classes = 5
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
logs = ''

print("Model: ViT")
model = models.vit_b_16(weights='IMAGENET1K_V1').to(device)

model.heads = nn.Sequential(
    nn.Linear(768, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 5)
)
model = model.to(device)
print(model)

labels = torch.tensor(all_dataset.targets)[train_dataset.indices]
class_counts = torch.bincount(labels)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()  # normalize
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
params = list(model.parameters())
#optimizer = optim.SGD(params, lr=1e-3, weight_decay=1e-6, momentum=0.9)#, eps=1e-8)
optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay = 1e-4)

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode='min', verbose=True)

accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='weighted').to(device)
confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true').to(device)
class_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"Number of trainable parameters in the model: {num_params}")

# Training
num_epochs = 30
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
        loss = F.cross_entropy(outputs, labels)
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
            loss = F.cross_entropy(outputs, labels)
            criterion(outputs, labels)
            val_loss += loss.item()
            outputs = F.softmax(outputs, dim=-1)
            val_accuracy = accuracy(outputs, labels)
            confmat.update(outputs, labels)
            val_class_accuracy = class_accuracy(outputs, labels)


    val_class_accuracy = class_accuracy.compute()

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = accuracy.compute()
    val_accuracies.append(float(val_accuracy))

    test_loss = val_loss


    test_accuracy = val_accuracy

    scheduler.step(val_loss)
    lr_log = f"LR: {optimizer.param_groups[0]['lr']}" # scheduler._last_lr
    print(lr_log)
    logs+=lr_log+'\n'

    train_results = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Training Accuracy: {train_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
    print(train_results)
    logs+=train_results+'\n'
    torch.save(model.state_dict(), os.path.join(output_dir, f'last.pth')) # Save model checkpoint

    if (epoch+1)%5==0: # Save every 5 epochs
        torch.save(model.state_dict(), os.path.join(output_dir, f'epoch{epoch+1}.pth'))


        fig, ax = plt.subplots()
        confmat_vals = np.around(confmat.compute().cpu().detach().numpy(), 3)
        im = ax.imshow(confmat_vals)

        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('True class')


        for i in range(num_classes):
            for j in range(num_classes):
                text = ax.text(j, i, confmat_vals[i, j],ha="center", va="center", color="black", fontsize=12)

        ax.set_title(f"Confusion Matrix on Test for epoch {epoch+1}")
        fig.savefig(os.path.join(output_dir, f"conf_mat_epoch{epoch+1}.png"))
        plt.close()


    if best_test_acc <= test_accuracy:
        best_epoch = epoch+1
        log = f"Improve accuracy from {best_test_acc} to {test_accuracy}"
        print(log)
        logs+=log+"\n"
        best_test_acc = test_accuracy
        torch.save(model.state_dict(), os.path.join(output_dir, f'best.pth'))


        fig, ax = plt.subplots()
        confmat_vals = np.around(confmat.compute().cpu().detach().numpy(), 3)
        im = ax.imshow(confmat_vals)

   
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('True class')

    
        for i in range(num_classes):
            for j in range(num_classes):
                text = ax.text(j, i, confmat_vals[i, j],ha="center", va="center", color="black", fontsize=12)

        ax.set_title("Confusion Matrix on Test for best model")
        fig.savefig(os.path.join(output_dir, "conf_mat_best.png"))
        plt.close()


    accuracy.reset(); class_accuracy.reset(); confmat.reset()


# Save to log file
with open(os.path.join(output_dir, 'log.txt'), 'w') as log_file:
    log_file.write(logs)
    log_file.write(f'Best val accuracy: {best_test_acc} in epoch {best_epoch}')
os.makedirs(output_dir, exist_ok=True)

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


# Save
torch.save(model, os.path.join('model_architecture.pth'))
torch.save(model.state_dict(), os.path.join(output_dir, f'model_weights.pth'))


# Testing
model.eval()

test_loss = 0
test_acc = 0
with torch.no_grad():
    for x, y in test_loader:
        x,y = x.to(device), y.to(device)
        pred = model(x)

        loss = criterion(pred, y)

        test_loss += loss.item()

        test_acc += accuracy(pred, y)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
print(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')

# Single batch testing w logits
model.eval()
with torch.no_grad():
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    print(outputs)
    print("Predicted classes", outputs.argmax(-1))
    print("Actual classes", labels)

print("Model complete") 
