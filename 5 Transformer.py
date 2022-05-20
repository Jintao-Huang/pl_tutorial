# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

# [Setup]

# Standard libraries
import math
import os
import urllib.request
from functools import partial
from urllib.error import HTTPError

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# PyTorch Lightning
import pytorch_lightning as pl
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

# Torchvision
import torchvision
# from IPython.display import set_matplotlib_formats
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm import tqdm  # .notebook

# plt.set_cmap("cividis")  # Set the default colormap

# %matplotlib inline
# set_matplotlib_formats("svg", "pdf")
matplotlib.rcParams["lines.linewidth"] = 2.0

sns.reset_orig()

#
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
CHECKPOINT_PATH = os.environ.get(
    "PATH_CHECKPOINT", "saved_models/Transformers/")

#
pl.seed_everything(42)

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
#
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial6/"
#
pretrained_files = ["ReverseTask.ckpt", "SetAnomalyTask.ckpt"]
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if "/" in file_name:
        os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print("Downloading %s..." % file_url)
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print(
                "Something went wrong. Please try to download the file manually,"
                " or contact the author with the full output including the following error:\n",
                e,
            )

# [The Transformer architecture]


# [What is attention]
# [Scaled dot product attention]
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        # mask == 0的地方 mask
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


seq_len, d_k = 3, 2
pl.seed_everything(42)
q = torch.randn(seq_len, d_k)
k = torch.randn(seq_len, d_k)
v = torch.randn(seq_len, d_k)
values, attention = scaled_dot_product(q, k, v)
print("Q\n", q)
print("K\n", k)
print("V\n", v)
print("Values\n", values)
print("Attention\n", attention)

# [Multi-head attention]


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):

        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length,
                          self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        #
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

# [Transformer encoder]


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.):
        super(EncoderBlock, self).__init__()

        #
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        #
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        #
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps

# [PE]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) / d_model * (-math.log(10000.0)))  # [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


encod_block = PositionalEncoding(d_model=512, max_len=512)
pe = encod_block.pe.squeeze().T.cpu().numpy()  # .t()

#
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
pos = ax.imshow(pe, origin="lower", extent=(0, pe.shape[1], 0, pe.shape[0]))
fig.colorbar(pos, ax=ax)
ax.set_xlabel("Position in sequence")
ax.set_ylabel("Hidden dimension")
ax.set_title("Positional encoding over hidden dimensions")
ax.set_xticks([i * 50 for i in range(0, pe.shape[1] // 50)])
ax.tick_params(axis='x', which='major', rotation=90)
ax.set_yticks([i * 50 for i in range(0, pe.shape[0] // 50)])
plt.show()
plt.close()


sns.set_theme()
fig, ax = plt.subplots(2, 2, figsize=(12, 4))
ax = [a for a_list in ax for a in a_list]
for i in range(len(ax)):
    ax[i].plot(np.arange(16), pe[i, :16], color="C%i" %
               i, marker="o", markersize=6, markeredgecolor="black")
    ax[i].set_title("Encoding in hidden dimension %i" % i)
    ax[i].set_xlabel("Position in sequence", fontsize=10)
    ax[i].set_ylabel("Positional encoding", fontsize=10)
    ax[i].set_xticks(np.arange(16))
    ax[i].tick_params(axis="both", which="major", labelsize=10)
    ax[i].tick_params(axis="both", which="minor", labelsize=8)
    ax[i].set_ylim(-1.2, 1.2)
fig.subplots_adjust(hspace=0.8)
sns.reset_orig()
plt.show()

# [LR warm-up]


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        # last_epoch=-1
        super(CosineWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


p = nn.Parameter(torch.empty(4, 4))
optimizer = optim.Adam([p], lr=1e-3)
lr_scheduler = CosineWarmupScheduler(
    optimizer=optimizer, warmup=100, max_iters=2000)

#
epochs = list(range(2000))
sns.set()
plt.figure(figsize=(8, 3))
# print([lr_scheduler.get_lr_factor(e) for e in epochs])
plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
plt.ylabel("Learning rate factor")
plt.xlabel("Iterations (in batches)")
plt.title("Cosine Warm-up Learning Rate Scheduler")
plt.show()
sns.reset_orig()


# [PyTorch Lightning Mudule]
class TransformerPredictor(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        lr,
        warmup,  # Usually between 50 and 500. [200]
        max_iters,
        dropout=0.,
        input_dropout=0.,
    ):

        super(TransformerPredictor, self).__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout), nn.Linear(
                self.hparams.input_dim, self.hparams.model_dim)
        )  # Position_wise
        self.positional_encoding = PositionalEncoding(
            d_model=self.hparams.model_dim)
        #
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_feedforward=2 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )
        #
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        # x: [Batch, SeqLen, input_dim]

        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super(TransformerPredictor, self).optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

# [Experiment]
# [Seq to Seq]


class ReverseDataset(data.Dataset):
    def __init__(self, num_categories, seq_len, size):
        super(ReverseDataset, self).__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size
        # [N, SeqLen]
        self.data = torch.randint(
            self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = torch.flip(inp_data, dims=(0,))
        return inp_data, labels


dataset = partial(ReverseDataset, 10, 16)
# shuffle, drop_last
num_workers = 0
train_loader = data.DataLoader(dataset(
    50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers)
val_loader = data.DataLoader(
    dataset(1000), batch_size=128, pin_memory=True, num_workers=num_workers)
test_loader = data.DataLoader(
    dataset(10000), batch_size=128, pin_memory=True, num_workers=num_workers)

inp_data, labels = train_loader.dataset[0]
print("Input data:", inp_data)
print("Labels:    ", labels)


class ReversePredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        inp_data, labels = batch
        inp_data = F.one_hot(
            inp_data, num_classes=self.hparams.num_classes).float()

        #
        preds = self.forward(inp_data, add_positional_encoding=True)
        loss = F.cross_entropy(
            preds.view(-1, preds.size(-1)), labels.view(-1))  # [N * SeqLen, C]
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        #
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)  # on_step=True, on_epoch=False
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


def train_reverse(**kwargs):
    root_dir = os.path.join(CHECKPOINT_PATH, "ReverseTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(save_weights_only=True,
                            mode="max", monitor="val_acc"),
            LearningRateMonitor("step")
        ],
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=10,
        gradient_clip_val=5,  # gradient_clip_algorithm='norm', 不改变梯度方向. 0.5-10之间
        log_every_n_steps = 10
        # progress_bar_refresh_rate=1,
    )
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = False

    #
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ReverseTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ReversePredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = ReversePredictor(
            max_iters=trainer.max_epochs * len(train_loader), **kwargs)
        trainer.fit(model, train_loader, val_loader)

    #
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(val_result, test_result)
    result = {"test_acc": test_result[0]["test_acc"],
              "val_acc": val_result[0]["test_acc"]}
    # fit/test会释放显存, 讲model转为cpu.
    model.to(device)
    return model, result


reverse_model, reverse_result = train_reverse(
    input_dim=train_loader.dataset.num_categories,
    model_dim=32,
    num_heads=1,
    num_classes=train_loader.dataset.num_categories,
    num_layers=1,
    dropout=0.0,
    lr=5e-4,
    warmup=50,
)


print("Val accuracy:  %4.2f%%" % (100. * reverse_result["val_acc"]))
print("Test accuracy: %4.2f%%" % (100. * reverse_result["test_acc"]))
#
data_input, labels = next(iter(val_loader))
inp_data = F.one_hot(
    data_input, num_classes=reverse_model.hparams.num_classes).float()
inp_data = inp_data.to(device)
attention_maps = reverse_model.get_attention_maps(inp_data)
# attention_maps: List[t:Tensor]; t: Tensor[128, 1, 16, 16]

print(len(attention_maps))
print(attention_maps[0].shape)
#


def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy()
                 for m in attn_maps]  # 每层取idx batch

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(
        num_heads * fig_size, num_layers * fig_size))  # Width, height
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(
                attn_maps[row][column], origin="upper", vmin=0)
            ax[row][column].set_xticks(
                list(range(seq_len)), input_data.tolist())
            ax[row][column].set_yticks(
                list(range(seq_len)), input_data.tolist())
            ax[row][column].set_title("Layer %i, Head %i" % (row, column))
    fig.subplots_adjust(hspace=0.5)
    plt.show()


# note: 每个横坐标进行softmax.
# note: x轴是输入, 不是label
plot_attention_maps(data_input, attention_maps, idx=0)


# [Set Anomaly Detection]
# ImageNet statistics
DATA_MEANS = np.array([0.485, 0.456, 0.406])
DATA_STD = np.array([0.229, 0.224, 0.225])
TORCH_DATA_MEANS = torch.from_numpy(DATA_MEANS).view(1, 3, 1, 1)
TORCH_DATA_STD = torch.from_numpy(DATA_STD).view(1, 3, 1, 1)

transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(),
     transforms.Normalize(DATA_MEANS, DATA_STD)]
)
train_set = CIFAR100(root=DATASET_PATH, train=True,
                     transform=transform, download=True)
test_set = CIFAR100(root=DATASET_PATH, train=False,
                    transform=transform, download=True)
#
# os.environ["TORCH_HOME"] = CHECKPOINT_PATH
pretrained_model = torchvision.models.resnet34(pretrained=True)
# In some models, it is called "fc", others have "classifier"
pretrained_model.fc = nn.Sequential()
pretrained_model.classifier = nn.Sequential()
pretrained_model = pretrained_model.to(device)

#
pretrained_model.eval()
# freeze. 用no_grad是不够的
for p in pretrained_model.parameters():
    p.requires_grad = False
#


@torch.no_grad()
def extract_features(dataset, save_file):
    if not os.path.isfile(save_file):
        data_loader = data.DataLoader(
            dataset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)
        extracted_features = []
        for imgs, _ in tqdm(data_loader):
            imgs = imgs.to(device)
            feats = pretrained_model(imgs)
            extracted_features.append(feats.cpu())
        extracted_features = torch.cat(extracted_features, dim=0)
        torch.save(extracted_features, save_file)
    else:
        extracted_features = torch.load(save_file)
    return extracted_features


train_feat_file = os.path.join(CHECKPOINT_PATH, "train_set_features.tar")
train_set_feats = extract_features(train_set, train_feat_file)

test_feat_file = os.path.join(CHECKPOINT_PATH, "test_set_features.tar")
test_feats = extract_features(test_set, test_feat_file)
#
print("Train:", train_set_feats.shape)  # [50000, 512]
print("Test: ", test_feats.shape)  # [10000, 512]

labels = train_set.targets
labels = torch.LongTensor(labels)
num_labels = labels.max() + 1
sorted_indices = torch.argsort(labels).reshape(
    num_labels, -1)  # [classes, num_imgs per class]

num_val_exmps = sorted_indices.shape[1] // 10

# train:val = 9:1
val_indices = sorted_indices[:, :num_val_exmps].reshape(-1)
train_indices = sorted_indices[:, num_val_exmps:].reshape(-1)

#
train_feats, train_labels = train_set_feats[train_indices], labels[train_indices]
val_feats, val_labels = train_set_feats[val_indices], labels[val_indices]
#


class SetAnomalyDataset(data.Dataset):
    def __init__(self, img_feats, labels, set_size=10, train=True):
        super().__init__()
        self.img_feats = img_feats  # [num_imgs, img_dim]
        self.labels = labels  # [num_imgs]
        self.set_size = set_size - 1
        self.train = train

        self.num_labels = labels.max() + 1
        self.img_idx_by_label = torch.argsort(
            self.labels).reshape(self.num_labels, -1)

        if not train:
            self.test_sets = self._create_test_sets()

    def _create_test_sets(self):
        test_sets = []
        num_imgs = self.img_feats.shape[0]
        np.random.seed(42)
        test_sets = [self.sample_img_set(self.labels[idx])
                     for idx in range(num_imgs)]
        test_sets = torch.stack(test_sets, dim=0)
        return test_sets

    def sample_img_set(self, anomaly_label):
        set_label = np.random.randint(self.num_labels - 1)
        if set_label >= anomaly_label:
            set_label += 1

        #
        img_indices = np.random.choice(
            self.img_idx_by_label.shape[1], size=self.set_size, replace=False)
        img_indices = self.img_idx_by_label[set_label, img_indices]
        return img_indices

    def __len__(self):
        return self.img_feats.shape[0]

    def __getitem__(self, idx):
        anomaly = self.img_feats[idx]
        if self.train:  # If train => sample
            img_indices = self.sample_img_set(self.labels[idx])
        else:  # If test => use pre-generated ones
            img_indices = self.test_sets[idx]

        #
        img_set = torch.cat(
            [self.img_feats[img_indices], anomaly[None]], dim=0)
        indices = torch.cat([img_indices, torch.LongTensor([idx])], dim=0)
        label = img_set.shape[0] - 1

        return img_set, indices, label


#
SET_SIZE = 10
test_labels = torch.LongTensor(test_set.targets)

train_anom_dataset = SetAnomalyDataset(
    train_feats, train_labels, set_size=SET_SIZE, train=True)
val_anom_dataset = SetAnomalyDataset(
    val_feats, val_labels, set_size=SET_SIZE, train=False)
test_anom_dataset = SetAnomalyDataset(
    test_feats, test_labels, set_size=SET_SIZE, train=False)

train_anom_loader = data.DataLoader(
    train_anom_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
)
val_anom_loader = data.DataLoader(
    val_anom_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
test_anom_loader = data.DataLoader(
    test_anom_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
# 
def visualize_exmp(indices, orig_dataset):
    images = [orig_dataset[idx][0] for idx in indices.reshape(-1)]  # 0: images, 1: labels
    images = torch.stack(images, dim=0)
    images = images * TORCH_DATA_STD + TORCH_DATA_MEANS
    # 这里的nrow与plt.subplots的ncols含义相同
    img_grid = torchvision.utils.make_grid(images, nrow=SET_SIZE, normalize=True, pad_value=0.5, padding=16)
    img_grid = img_grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 8))
    plt.title("Anomaly examples on CIFAR100")
    plt.imshow(img_grid)
    plt.axis("off")
    plt.show()
    plt.close()



_, indices, _ = next(iter(test_anom_loader))
visualize_exmp(indices[:4], test_set)  # [4, 10], [N, H, W, C]
# 
class AnomalyPredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        img_sets, _, labels = batch
        # 
        preds = self.forward(img_sets, add_positional_encoding=False)
        preds = preds.squeeze(dim=-1)  # Shape: [Batch_size, set_size]
        # labels: [Batch_size]
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("%s_loss" % mode, loss, prog_bar=True)
        self.log("%s_acc" % mode, acc, prog_bar=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")

# 

def train_anomaly(**kwargs):
    # 
    root_dir = os.path.join(CHECKPOINT_PATH, "SetAnomalyTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=100,
        gradient_clip_val=2,
        log_every_n_steps=10
        # progress_bar_refresh_rate=1,
    )
    trainer.logger._default_hp_metric = False  # Optional logging argument that we don't need

    # 
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SetAnomalyTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = AnomalyPredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = AnomalyPredictor(max_iters=trainer.max_epochs * len(train_anom_loader), **kwargs)
        trainer.fit(model, train_anom_loader, val_anom_loader)
        model = AnomalyPredictor.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # 
    train_result = trainer.test(model, dataloaders=train_anom_loader, verbose=False)
    val_result = trainer.test(model, dataloaders=val_anom_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_anom_loader, verbose=False)
    result = {
        "test_acc": test_result[0]["test_acc"],
        "val_acc": val_result[0]["test_acc"],
        "train_acc": train_result[0]["test_acc"],
    }

    model = model.to(device)
    return model, result


anomaly_model, anomaly_result = train_anomaly(
    input_dim=train_anom_dataset.img_feats.shape[-1],
    model_dim=256,
    num_heads=4,
    num_classes=1,
    num_layers=4,
    dropout=0.1,
    input_dropout=0.1,
    lr=5e-4,
    warmup=100,
)
print("Train accuracy: %4.2f%%" % (100. * anomaly_result["train_acc"]))
print("Val accuracy:   %4.2f%%" % (100. * anomaly_result["val_acc"]))
print("Test accuracy:  %4.2f%%" % (100. * anomaly_result["test_acc"]))

# 
inp_data, indices, labels = next(iter(test_anom_loader))
inp_data = inp_data.to(device)

anomaly_model.eval()

with torch.no_grad():
    preds = anomaly_model.forward(inp_data, add_positional_encoding=False)
    preds = F.softmax(preds.squeeze(dim=-1), dim=-1)

    # 
    permut = np.random.permutation(inp_data.shape[1])
    perm_inp_data = inp_data[:, permut]
    perm_preds = anomaly_model.forward(perm_inp_data, add_positional_encoding=False)
    perm_preds = F.softmax(perm_preds.squeeze(dim=-1), dim=-1)

assert (preds[:, permut] - perm_preds).abs().max() < 1e-3, "Predictions are not permutation equivariant"

print("Preds\n", preds[0, permut].cpu().numpy())
print("Permuted preds\n", perm_preds[0].cpu().numpy())
# 
attention_maps = anomaly_model.get_attention_maps(inp_data, add_positional_encoding=False)
predictions = preds.argmax(dim=-1)

def visualize_prediction(idx):
    # indices, test_set, predictions, attention_maps
    visualize_exmp(indices[idx : idx + 1], test_set)
    print("Prediction:", predictions[idx].item())
    plot_attention_maps(input_data=None, attn_maps=attention_maps, idx=idx)


visualize_prediction(0)
# 
mistakes = torch.where(predictions != 9)[0].cpu().numpy()
print("Indices with mistake:", mistakes)

visualize_prediction(mistakes[-1])
print("Probabilities:")
for i, p in enumerate(preds[mistakes[-1]].cpu().numpy()):
    print("Image %i: %4.2f%%" % (i, 100.0 * p))

# [Conclusion]
