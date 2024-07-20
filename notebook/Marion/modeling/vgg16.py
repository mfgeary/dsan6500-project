import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import math
import os

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

random_seed = 1989
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if device == "cuda":
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
elif device == "mps":
    torch.mps.manual_seed(random_seed)
    torch.backends.mps.deterministic = True
    torch.backends.mps.benchmark = False

# Read in the data
train_label = pd.read_csv("../../data/train_label_coordinates_preprocessed_split.csv")
labels = train_label["condition"]
# Convert labels to integers
label_map = {
    "Spinal Canal Stenosis": 0,
    "Right Neural Foraminal Narrowing": 1,
    "Left Neural Foraminal Narrowing": 2,
    "Right Subarticular Stenosis": 3,
    "Left Subarticular Stenosis": 4
}
labels = labels.map(label_map)

split_set = train_label["dataset"]

path_to_image = train_label["preprocessed_file_path"]

# Create data loader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, path_to_image, labels, split_set, transform=None):
        self.path_to_image = path_to_image
        self.labels = labels
        self.transform = transform
        self.split_set = split_set

    def __len__(self):
        return len(self.path_to_image)

    def __getitem__(self, idx):
        path_starter = "../../data/"
        image = Image.open(path_starter + self.path_to_image[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_split_set(self):
        return self.split_set
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # convert from grayscale to RGB
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor()
])

batch_size = 64

# Get the train data, where split_set == "train"
train_data = ImageDataset(path_to_image, labels, split_set, transform)
train_data_indices = [i for i in range(len(train_data)) if train_data.get_split_set()[i] == "train"]
train_data = torch.utils.data.Subset(train_data, train_data_indices)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Get the validation data, where split_set == "val"
val_data = ImageDataset(path_to_image, labels, split_set, transform)
val_data_indices = [i for i in range(len(val_data)) if val_data.get_split_set()[i] == "val"]
val_data = torch.utils.data.Subset(val_data, val_data_indices)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Get the test data, where split_set == "test"
test_data = ImageDataset(path_to_image, labels, split_set, transform)
test_data_indices = [i for i in range(len(test_data)) if test_data.get_split_set()[i] == "test"]
test_data = torch.utils.data.Subset(test_data, test_data_indices)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

from sklearn import metrics as sklearn_metrics

def train(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, "Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = loss_fn(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)

    return epoch_loss

def get_metrics(labels, predictions, probs):
    metrics = {
        "Accuracy": sklearn_metrics.accuracy_score(labels, predictions),
        "Top-2 Accuracy": sklearn_metrics.top_k_accuracy_score(labels, probs, k=2),
        "Top-3 Accuracy": sklearn_metrics.top_k_accuracy_score(labels, probs, k=3),
        "Macro Recall": sklearn_metrics.recall_score(labels, predictions, average='macro'),
        "Weighted Recall": sklearn_metrics.recall_score(labels, predictions, average='weighted'),
        "Macro F1": sklearn_metrics.f1_score(labels, predictions, average='macro'),
        "Weighted F1": sklearn_metrics.f1_score(labels, predictions, average='weighted'),
        "Weighted AUC Score": sklearn_metrics.roc_auc_score(labels, probs, average='weighted', multi_class='ovr'),
    }
    return metrics


def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, "Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = loss_fn(outputs, labels)
            
            running_loss += batch_loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predictions == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    metrics = get_metrics(all_labels, all_predictions, all_probs)
    accuracy = correct_predictions / total_predictions * 100

    return epoch_loss, accuracy, metrics

class EarlyStopping():
    def __init__(self, patience = 2, verbose = False, delta = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = None
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            if self.best_loss is not None:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(f'Saving model with validation loss: {val_loss:.6f}')

        self.best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

print("Using VGG16 model on {}.".format(device))
# Define the model
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 5)
model = model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 30
patience = 3
early_stopping = EarlyStopping(patience = patience, verbose = True)

train_losses = []
val_losses = []
val_accuracies = []
metrics_list = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, loss, optimizer, device)
    val_loss, val_accuracy, metrics = evaluate(model, val_loader, loss, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    metrics_list.append(metrics)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print('Early stopping')
        break

# save the metrics_list
torch.save(metrics_list, "metrics_list_vgg16_0001.pt")
# save the train_losses, val_losses, val_accuracies as csv
df = pd.DataFrame({
    "train_loss": train_losses,
    "val_loss": val_losses,
    "val_accuracy": val_accuracies
})

df.to_csv("losses_accuracies_vgg16_0001.csv", index=False)

# evaluate on test set
model.load_state_dict(torch.load('best_model.pt'))

test_loss, test_accuracy, test_metrics = evaluate(model, test_loader, loss, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
print(test_metrics)

# Plot the training and validation losses
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('vgg16_loss_0001.png')
plt.show()

# Plot the validation accuracy
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('vgg16_val_accuracy_0001.png')
plt.show()