# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# [The Basics of PyTorch]
print("Using torch", torch.__version__)
torch.manual_seed(42)
# [Tensors]
x = torch.Tensor(2, 3, 4)
print(x)
# [Initialization]
x = torch.Tensor([[1, 2], [3, 4]])
print(x)

x = torch.rand(2, 3, 4)
print(x)

shape = x.shape
print("Shape:", shape)

size = x.size()
print("Size:", size)

dim1, dim2, dim3 = x.size()
print("Size:", dim1, dim2, dim3)
# [Tensor to Numpy, and Numpy to Tensor]
np_arr = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(np_arr)

print("Numpy array:", np_arr)
print("PyTorch tensor:", tensor)

tensor = torch.arange(4)
np_arr = tensor.numpy()

print("PyTorch tensor:", tensor)
print("Numpy array:", np_arr)

np_arr = tensor.cpu().numpy()
# [Operations]
x1 = torch.rand(2, 3)
x2 = torch.rand(2, 3)
y = x1 + x2

print("X1", x1)
print("X2", x2)
print("Y", y)

x1 = torch.rand(2, 3)
x2 = torch.rand(2, 3)
print("X1 (before)", x1)
print("X2 (before)", x2)

x2.add_(x1)
print("X1 (after)", x1)
print("X2 (after)", x2)

x = torch.arange(6)
print("X", x)

x = x.view(2, 3)
print("X", x)

x = x.permute(1, 0)
print("X", x)

x = torch.arange(6)
x = x.view(2, 3)
print("X", x)

W = torch.arange(9).view(3, 3)
print("W", W)

h = torch.matmul(x, W)
print("h", h)
# [Indexing]
x = torch.arange(12).view(3, 4)
print("X", x)

print(x[:, 1])
print(x[0])
print(x[:2, -1])
print(x[1:3, :])
# [Dynamic Computation Graph and Backpropagation]
x = torch.ones((3,))
print(x.requires_grad)

x.requires_grad_(True)
print(x.requires_grad)

x = torch.arange(3, dtype=torch.float32, requires_grad=True)
print("X", x)

a = x + 2
b = a ** 2
c = b + 3
y = c.mean()
print("Y", y)

y.backward()
print(x.grad)

# [GPU support]
gpu_avail = torch.cuda.is_available()
print(f"Is the GPU available? {gpu_avail}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

x = torch.zeros(2, 3)
x = x.to(device)
print("X", x)

x = torch.randn(5000, 5000)

start_time = time.time()
_ = torch.matmul(x, x)
end_time = time.time()
print(f"CPU time: {(end_time - start_time):6.5f}s")

if torch.cuda.is_available():
    x = x.to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = torch.matmul(x, x)
    end.record()
    torch.cuda.synchronize()
    print(f"GPU time: {0.001 * start.elapsed_time(end):6.5f}s")

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
# [Learning by example: Continuous XOR]
# [The model]
from torch import nn
import torch.nn.functional as F


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, x):
        pass


# [Simple classifier]
class SimpleClassifier(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(SimpleClassifier, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
print(model)

for name, param in model.named_parameters():
    print(f"Parameter {name}, shape {param.shape}")

# [The data]
from torch.utils import data


# [The dataset class]
class XORDataset(data.Dataset):
    def __init__(self, size, std=0.1):
        super(XORDataset, self).__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)  # [x, y]
        label = (data.sum(dim=1) == 1).to(torch.long)
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label


dataset = XORDataset(size=200)
print("Size of dataset:", len(dataset))
print("Data point 0:", dataset[0])


def visualize_samples(data, label):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()


visualize_samples(dataset.data, dataset.label)
plt.show()
# [The data loader class]
data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
data_inputs, data_labels = next(iter(data_loader))
print("Data inputs", data_inputs.shape, "\n", data_inputs)
print("Data labels", data_labels.shape, "\n", data_labels)
# [Optimization]
# [Loss modules]
# [^数值稳定性]
x = torch.Tensor([100, 100.])
x.requires_grad_(True)
x2 = torch.sigmoid(x)
y = torch.Tensor([0, 0.])
loss = nn.BCELoss()
z = loss(x2, y)
z.backward()
print(z, x.grad)
#
x = torch.Tensor([100, 100.])
x.requires_grad_(True)
y = torch.Tensor([0, 0.])
loss = nn.BCEWithLogitsLoss()
z = loss(x, y)
z.backward()
print(z, x.grad)
#
loss_module = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# [Training]
train_dataset = XORDataset(size=1000)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

model.to(device)
from tqdm import tqdm


def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)  # Output is [Batch size, 1], but we want [Batch size]

            loss = loss_module(preds, data_labels.float())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


train_model(model, optimizer, train_data_loader, loss_module)
# [Saving a model]
# [^test state_dict]
state_dict = model.state_dict()
print(state_dict)
model.register_buffer("a", torch.Tensor([1, 2, 3.]))
print(list(model.buffers()))
print(model.state_dict())
del model.a
#
state_dict = model.state_dict()
print(state_dict)
torch.save(state_dict, "our_model.tar")
state_dict = torch.load("our_model.tar")
new_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
new_model.load_state_dict(state_dict)
print("Original model\n", model.state_dict())
print("\nLoaded model\n", new_model.state_dict())
# [Evaluation]


test_dataset = XORDataset(size=500)
test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)


def eval_model(model, data_loader):
    model.eval()
    true_preds, num_preds = 0.0, 0.0

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)
            pred_labels = (preds >= 0.5).long()

            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0 * acc:4.2f}%")


eval_model(model, test_data_loader)

# [Visualizing classification boundaries]
from matplotlib.colors import to_rgba, to_rgb


@torch.no_grad()
def visualize_classification(model, data, label):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = plt.figure(figsize=(4, 4), dpi=500)
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()
    #
    model.to(device)
    c0 = torch.Tensor(to_rgb("C0")).to(device)
    c1 = torch.Tensor(to_rgb("C1")).to(device)
    x1 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    x2 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='xy')
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds = model(model_inputs)  # torch.Size([200, 200, 1])
    preds = torch.sigmoid(preds)
    output_image = (1 - preds) * c0[None, None] + preds * c1[None, None]  # [200, 200, 3]
    output_image = output_image.cpu().numpy()

    plt.imshow(output_image, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    return fig


visualize_classification(model, dataset.data, dataset.label)
plt.show()

#
import os

os.remove("our_model.tar")

#
from torch.utils.tensorboard import SummaryWriter
def train_model_with_logger(model, optimizer, data_loader, loss_module, val_dataset, num_epochs=100, logging_dir='runs/our_experiment'):
    writer = SummaryWriter(logging_dir)
    model_plotted = False
    # writer.add_text("config", "...")
    #
    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.
        for data_inputs, data_labels in data_loader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            if not model_plotted:
                writer.add_graph(model, data_inputs)
                model_plotted = True

            #
            preds = model(data_inputs)   # [Batch size, 1]
            preds = preds.squeeze(dim=1)
            loss = loss_module(preds, data_labels.float())
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            epoch_loss += loss.item()

        #
        epoch_loss /= len(data_loader)
        writer.add_scalar('training_loss',
                          epoch_loss,
                          global_step = epoch + 1)

        #
        if (epoch + 1) % 10 == 0:
            fig = visualize_classification(model, val_dataset.data, val_dataset.label)
            writer.add_figure(f'predictions',
                              fig,
                              global_step = epoch + 1)

    writer.close()

model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
train_model_with_logger(model, optimizer, train_data_loader, loss_module, val_dataset=dataset)
# tensorboard --logdir runs/our_experiment
# tensorboard --logdir runs
