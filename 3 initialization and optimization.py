# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import copy
import json
import math
import os
import urllib.request
from urllib.error import HTTPError

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

# sns.set()

# [Setup]
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/InitOptim/")

pl.seed_everything(42)

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)
#
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial4/"
# Files to download
pretrained_files = [
    "FashionMNIST_SGD.config",
    "FashionMNIST_SGD_results.json",
    "FashionMNIST_SGD.tar",
    "FashionMNIST_SGDMom.config",
    "FashionMNIST_SGDMom_results.json",
    "FashionMNIST_SGDMom.tar",
    "FashionMNIST_Adam.config",
    "FashionMNIST_Adam_results.json",
    "FashionMNIST_Adam.tar",
]
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print(
                "Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
                e,
            )
# [Preparation]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))])

train_dataset = FashionMNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

test_set = FashionMNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

train_loader = data.DataLoader(train_set, batch_size=1024, shuffle=True, drop_last=False)
val_loader = data.DataLoader(val_set, batch_size=1024, shuffle=False, drop_last=False)
test_loader = data.DataLoader(test_set, batch_size=1024, shuffle=False, drop_last=False)

print("Mean", (train_dataset.data.float() / 255.0).mean().item())
print("Std", (train_dataset.data.float() / 255.0).std().item())

imgs, _ = next(iter(train_loader))
print(f"Mean: {imgs.mean().item():5.3f}")
print(f"Standard deviation: {imgs.std().item():5.3f}")
print(f"Maximum: {imgs.max().item():5.3f}")
print(f"Minimum: {imgs.min().item():5.3f}")


#
class BaseNetwork(nn.Module):
    def __init__(self, act_fn, input_size=784, num_classes=10, hidden_sizes=[512, 256, 256, 128]):

        super(BaseNetwork, self).__init__()

        #
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]), act_fn]
        layers += [nn.Linear(layer_sizes[-1], num_classes)]
        #
        self.layers = nn.ModuleList(layers)

        self.config = {
            "act_fn": act_fn.__class__.__name__,
            "input_size": input_size,
            "num_classes": num_classes,
            "hidden_sizes": hidden_sizes,
        }

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x


class Identity(nn.Module):
    # nn.Identity
    def forward(self, x):
        return x


act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "identity": Identity}


#
def plot_dists(val_dict, color="C0", xlabel=None, stat="count", use_kde=True):
    columns = len(val_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 3, 2.5))

    for fig_index, key in enumerate(val_dict.keys()):
        key_ax = ax[fig_index % columns]
        sns.histplot(
            val_dict[key],
            ax=key_ax,
            color=color,
            bins=50,
            stat=stat,
            # [?]
            kde=use_kde and ((val_dict[key].max() - val_dict[key].min()) > 1e-8),
        )
        hidden_dim_str = (
            r"(%d $\to$ %d)" % (val_dict[key].shape[1], val_dict[key].shape[0]) if len(val_dict[key].shape) > 1 else ""
        )
        key_ax.set_title(f"{key} {hidden_dim_str}")
        if xlabel is not None:
            key_ax.set_xlabel(xlabel)
    fig.subplots_adjust(wspace=0.4)
    return fig


def visualize_weight_distribution(model, color="C0"):
    weights = {}
    for name, param in model.named_parameters():
        if name.endswith(".bias"):  # layers.4.weight
            continue
        key_name = f"Layer {name.split('.')[1]}"
        weights[key_name] = param.detach().view(-1).cpu().numpy()

    #
    fig = plot_dists(weights, color=color, xlabel="Weight vals")
    fig.suptitle("Weight distribution", fontsize=14, y=0.98)
    plt.show()
    plt.close()


def visualize_gradients(model, color="C0", print_variance=False):
    # w的grad, 而不是x的grad
    model.eval()
    small_loader = data.DataLoader(train_set, batch_size=1024, shuffle=False)
    imgs, labels = next(iter(small_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    model.zero_grad()
    preds = model(imgs)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    grads = {
        name: params.grad.clone().view(-1).cpu().numpy()
        for name, params in model.named_parameters()
        if "weight" in name
    }
    # model.zero_grad()

    #
    fig = plot_dists(grads, color=color, xlabel="Grad magnitude")
    fig.suptitle("Gradient distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()

    if print_variance:
        for key in sorted(grads.keys()):
            print(f"{key} - Variance: {np.var(grads[key])}")


def visualize_activations(model, color="C0", print_variance=False):
    model.eval()
    small_loader = data.DataLoader(train_set, batch_size=1024, shuffle=False)
    imgs, labels = next(iter(small_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    #
    feats = imgs.view(imgs.shape[0], -1)
    activations = {}
    with torch.no_grad():
        for layer_index, layer in enumerate(model.layers):
            feats = layer(feats)
            if isinstance(layer, nn.Linear):
                activations[f"Layer {layer_index}"] = feats.view(-1).cpu().numpy()

    #
    fig = plot_dists(activations, color=color, stat="density", xlabel="Activation vals")
    fig.suptitle("Activation distribution", fontsize=14, y=0.98)
    plt.show()
    plt.close()

    if print_variance:
        for key in sorted(activations.keys()):
            print(f"{key} - Variance: {np.var(activations[key])}")


# [Initialization]
model = BaseNetwork(act_fn=Identity()).to(device)


# [Constant initialization]
def const_init(model, fill=0.0):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.fill_(fill)


const_init(model, fill=0.005)
visualize_gradients(model)
visualize_activations(model, print_variance=True)


# [Constant variance]
def var_init(model, std=0.01):
    for name, param in model.named_parameters():
        param.data.normal_(mean=0.0, std=std)


var_init(model, std=0.01)
visualize_activations(model, print_variance=True)

var_init(model, std=0.1)
visualize_activations(model, print_variance=True)


# [How to find appropriate initialization values]
def equal_var_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            param.data.normal_(std=1.0 / math.sqrt(param.shape[1]))


equal_var_init(model)
visualize_weight_distribution(model)
visualize_activations(model, print_variance=True)


#
def xavier_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
            param.data.uniform_(-bound, bound)


xavier_init(model)
visualize_gradients(model, print_variance=True)
visualize_activations(model, print_variance=True)

model = BaseNetwork(act_fn=nn.Tanh()).to(device)
xavier_init(model)
visualize_gradients(model, print_variance=True)
visualize_activations(model, print_variance=True)


#
def kaiming_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.startswith("layers.0"):  # The first layer does not have ReLU applied on its input
            param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
        else:
            param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))


model = BaseNetwork(act_fn=nn.ReLU()).to(device)
kaiming_init(model)
visualize_gradients(model, print_variance=True)
visualize_activations(model, print_variance=True)


# [test torch.nn.init.calculate_gain]


# [Optimization]
def _get_config_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".config")


def _get_model_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".tar")


def _get_result_file(model_path, model_name):
    return os.path.join(model_path, model_name + "_results.json")


def load_model(model_path, model_name, net=None):
    config_file = _get_config_file(model_path, model_name)
    model_file = _get_model_file(model_path, model_name)
    assert os.path.isfile(
        config_file
    ), f'Could not find the config file "{config_file}". Are you sure this is the correct path and you have your model config stored here?'
    assert os.path.isfile(
        model_file
    ), f'Could not find the model file "{model_file}". Are you sure this is the correct path and you have your model stored here?'
    with open(config_file) as f:
        config_dict = json.load(f)
    if net is None:
        act_fn_name = config_dict.pop("act_fn").lower()
        assert (
                act_fn_name in act_fn_by_name
        ), f'Unknown activation function "{act_fn_name}". Please add it to the "act_fn_by_name" dict.'
        act_fn = act_fn_by_name[act_fn_name]()
        net = BaseNetwork(act_fn=act_fn, **config_dict)
    net.load_state_dict(torch.load(model_file))
    return net


def save_model(model, model_path, model_name):
    config_dict = model.config
    os.makedirs(model_path, exist_ok=True)
    config_file = _get_config_file(model_path, model_name)
    model_file = _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)


def train_model(net, model_name, optim_func, max_epochs=50, batch_size=256, overwrite=False):
    file_exists = os.path.isfile(_get_model_file(CHECKPOINT_PATH, model_name))
    if file_exists and not overwrite:
        print(f'Model file of "{model_name}" already exists. Skipping training...')
        with open(_get_result_file(CHECKPOINT_PATH, model_name)) as f:
            results = json.load(f)
    else:
        if file_exists:
            print("Model file exists, but will be overwritten...")

        #
        optimizer = optim_func(net.parameters())
        loss_module = nn.CrossEntropyLoss()
        train_loader_local = data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True
        )

        results = None
        val_scores = []
        train_losses, train_scores = [], []
        best_val_epoch = -1
        for epoch in range(max_epochs):
            train_acc, val_acc, epoch_losses = epoch_iteration(
                net, loss_module, optimizer, train_loader_local, val_loader, epoch
            )
            train_scores.append(train_acc)
            val_scores.append(val_acc)
            train_losses += epoch_losses

            if len(val_scores) == 1 or val_acc > val_scores[best_val_epoch]:
                print("\t   (New best performance, saving model...)")
                save_model(net, CHECKPOINT_PATH, model_name)
                best_val_epoch = epoch

    if results is None:
        load_model(CHECKPOINT_PATH, model_name, net=net)
        test_acc = test_model(net, test_loader)
        results = {
            "test_acc": test_acc,
            "val_scores": val_scores,
            "train_losses": train_losses,
            "train_scores": train_scores,
        }
        with open(_get_result_file(CHECKPOINT_PATH, model_name), "w") as f:
            json.dump(results, f)

    #
    plt.plot([i for i in range(1, len(results["train_scores"]) + 1)], results["train_scores"], label="Train")
    plt.plot([i for i in range(1, len(results["val_scores"]) + 1)], results["val_scores"], label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Validation accuracy")
    plt.ylim(min(results["val_scores"]), max(results["train_scores"]) * 1.01)
    plt.title(f"Validation performance of {model_name}")
    plt.legend()
    plt.show()
    plt.close()

    print((f" Test accuracy: {results['test_acc'] * 100:4.2f}% ").center(50, "=") + "\n")
    return results


def epoch_iteration(net, loss_module, optimizer, train_loader_local, val_loader, epoch):
    #
    net.train()
    true_preds, count = 0., 0
    epoch_losses = []
    t = tqdm(train_loader_local)
    for imgs, labels in t:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = net(imgs)
        loss = loss_module(preds, labels)
        loss.backward()
        optimizer.step()
        #
        true_preds += (preds.argmax(dim=-1) == labels).sum().item()
        count += labels.shape[0]
        t.set_description(f"Epoch {epoch + 1}: loss={loss.item():4.2f}")
        epoch_losses.append(loss.item())
    train_acc = true_preds / count

    #
    val_acc = test_model(net, val_loader)
    print(
        f"[Epoch {epoch + 1:2d}] Training accuracy: {train_acc * 100:05.2f}%, Validation accuracy: {val_acc * 100:05.2f}%"
    )
    return train_acc, val_acc, epoch_losses


def test_model(net, data_loader):
    net.eval()
    true_preds, count = 0., 0
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            preds = net(imgs).argmax(dim=-1)
            true_preds += (preds == labels).sum().item()
            count += labels.shape[0]
    test_acc = true_preds / count
    return test_acc


#

class OptimizerTemplate:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()  # For second-order optimizers important. (Hessian)
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            self.update_param(p)

    def update_param(self, p):
        raise NotImplementedError


class SGD(OptimizerTemplate):
    def __init__(self, params, lr):
        super(SGD, self).__init__(params, lr)

    def update_param(self, p):
        p_update = -self.lr * p.grad
        p.add_(p_update)


class SGDMomentum(OptimizerTemplate):
    def __init__(self, params, lr, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.param_momentum = {p: torch.zeros_like(p) for p in self.params}

    def update_param(self, p):
        self.param_momentum[p] = (1 - self.momentum) * p.grad + self.momentum * self.param_momentum[p]
        p_update = -self.lr * self.param_momentum[p]
        p.add_(p_update)


class Adam(OptimizerTemplate):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.param_step = {p: 0 for p in self.params}  # "t" for each parameter for bias correction
        self.param_momentum = {p: torch.zeros_like(p) for p in self.params}
        self.param_2nd_momentum = {p: torch.zeros_like(p) for p in self.params}

    def update_param(self, p):
        self.param_step[p] += 1

        self.param_momentum[p] = (1 - self.beta1) * p.grad + self.beta1 * self.param_momentum[p]
        self.param_2nd_momentum[p] = (1 - self.beta2) * p.grad ** 2 + self.beta2 * self.param_2nd_momentum[p]

        bias_correction_1 = 1 - self.beta1 ** self.param_step[p]  # ?可优化
        bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

        p_mom = self.param_momentum[p] / bias_correction_1
        p_2nd_mom = self.param_2nd_momentum[p] / bias_correction_2

        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr * p_mom

        p.add_(p_update)


# [Comparing optimizers on model training]
base_model = BaseNetwork(act_fn=nn.ReLU(), hidden_sizes=[512, 256, 256, 128])
kaiming_init(base_model)

SGD_model = copy.deepcopy(base_model).to(device)
SGD_results = train_model(
    SGD_model,
    "FashionMNIST_SGD",
    lambda params: SGD(params, lr=1e-1),
    max_epochs=40,
    batch_size=256
)

SGDMom_model = copy.deepcopy(base_model).to(device)
SGDMom_results = train_model(
    SGDMom_model,
    "FashionMNIST_SGDMom",
    lambda params: SGDMomentum(params, lr=1e-1, momentum=0.9),
    max_epochs=40,
    batch_size=256,
)

Adam_model = copy.deepcopy(base_model).to(device)
Adam_results = train_model(
    Adam_model,
    "FashionMNIST_Adam",
    lambda params: Adam(params, lr=1e-3),
    max_epochs=40,
    batch_size=256
)
# [Pathological curvatures]
from matplotlib import cm


def pathological_curve_loss(w1, w2):
    x1_loss = torch.tanh(w1) ** 2 + 0.01 * torch.abs(w1)
    x2_loss = torch.sigmoid(w2)
    return x1_loss + x2_loss


def plot_curve(
        curve_fn, x_range=(-5, 5), y_range=(-5, 5), plot_3d=False, cmap=cm.viridis, title="Pathological curvature"
):
    ax = plt.subplot(projection="3d") if plot_3d else plt.subplot()

    x = torch.linspace(x_range[0], x_range[1], 100)
    y = torch.linspace(y_range[0], y_range[1], 100)
    x, y = torch.meshgrid([x, y], indexing='xy')
    z = curve_fn(x, y)
    x, y, z = x.numpy(), y.numpy(), z.numpy()

    if plot_3d:
        ax.plot_surface(x, y, z, cmap=cmap)
        ax.set_zlabel("loss")
    else:
        # ax.imshow(z, cmap=cmap, extent=(x_range[0], x_range[1], y_range[1], y_range[0]))
        ax.imshow(z, cmap=cmap, origin='lower', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    plt.title(title)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    plt.tight_layout()
    return ax


sns.reset_orig()
_ = plot_curve(pathological_curve_loss, plot_3d=True)
plt.show()
_ = plot_curve(pathological_curve_loss, plot_3d=False)
plt.show()


#
def train_curve(optimizer_func, curve_func=pathological_curve_loss, num_updates=100, init=[5, 5]):
    weights = nn.Parameter(torch.FloatTensor(init))
    optim = optimizer_func([weights])  # 只有一层

    list_points = []
    for _ in range(num_updates):
        loss = curve_func(weights[0], weights[1])
        list_points.append(torch.stack([*weights.detach(), loss.detach()], dim=0))
        optim.zero_grad()
        loss.backward()
        optim.step()
    points = torch.stack(list_points, dim=0).numpy()
    return points


SGD_points = train_curve(lambda params: SGD(params, lr=10))
SGDMom_points = train_curve(lambda params: SGDMomentum(params, lr=10, momentum=0.9))
Adam_points = train_curve(lambda params: Adam(params, lr=1))

all_points = np.concatenate([SGD_points, SGDMom_points, Adam_points], axis=0)
ax = plot_curve(
    pathological_curve_loss,
    x_range=(-np.abs(all_points[:, 0]).max(), np.abs(all_points[:, 0]).max()),
    y_range=(all_points[:, 1].min(), all_points[:, 1].max()),
    plot_3d=False,
)
ax.plot(SGD_points[:, 0], SGD_points[:, 1], marker="o", linestyle='-', color="red", label="SGD")
ax.plot(SGDMom_points[:, 0], SGDMom_points[:, 1], marker="o", linestyle='-', color="blue", label="SGDMom")
ax.plot(Adam_points[:, 0], Adam_points[:, 1], marker="o", linestyle='-', color="grey", label="Adam")
plt.legend()
plt.show()


# [Steep optima]

def bivar_gaussian(w1, w2, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
    norm = 1 / (2 * torch.pi * x_sig * y_sig)
    x_exp = (w1 - x_mean) ** 2 / (x_sig ** 2)
    y_exp = (w2 - y_mean) ** 2 / (y_sig ** 2)
    return norm * torch.exp(-1 / 2 * (x_exp + y_exp))


def comb_func(w1, w2):
    z = -bivar_gaussian(w1, w2, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.5)
    z -= bivar_gaussian(w1, w2, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.5)
    z -= bivar_gaussian(w1, w2, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.5)
    return z


_ = plot_curve(comb_func, x_range=(-2, 2), y_range=(-2, 2), plot_3d=True, title="Steep optima")
plt.show()

SGD_points = train_curve(lambda params: SGD(params, lr=0.5), comb_func, init=[0, 0])
SGDMom_points = train_curve(lambda params: SGDMomentum(params, lr=1, momentum=0.9), comb_func, init=[0, 0])
Adam_points = train_curve(lambda params: Adam(params, lr=0.2), comb_func, init=[0, 0])

all_points = np.concatenate([SGD_points, SGDMom_points, Adam_points], axis=0)
ax = plot_curve(comb_func, x_range=(-2, 2), y_range=(-2, 2), plot_3d=False, title="Steep optima")
ax.plot(SGD_points[:, 0], SGD_points[:, 1], marker="o", linestyle='-', color="red", label="SGD", alpha=0.2)
ax.plot(SGDMom_points[:, 0], SGDMom_points[:, 1], marker="o", linestyle='-', color="blue", label="SGDMom", alpha=0.2)
ax.plot(Adam_points[:, 0], Adam_points[:, 1], marker="o", linestyle='-', color="grey", label="Adam", alpha=0.2)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.legend()
plt.show()

# [What optimizer to take]
