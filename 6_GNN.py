# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

# [Setup]


import torch_geometric.datasets as geom_datasets
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
import torch_geometric.loader as geom_loader
import os

import urllib.request
from urllib.error import HTTPError

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from pytorch_lightning.callbacks import ModelCheckpoint

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
#
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")

#
pl.seed_everything(42)

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial7/"
pretrained_files = ["NodeLevelMLP.ckpt",
                    "NodeLevelGNN.ckpt", "GraphLevelGraphConv.ckpt"]

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
                "Something went wrong. Please try to download the file from the GDrive folder,"
                " or contact the author with the full output including the following error:\n",
                e,
            )

# [Graph Neural Networks]
# [Graph representation]
# [Graph Convolutions]

# H_l: [N_Nodes, F]
# W_l: [F, F]
# A: [N_Nodes, N_Nodes]
# D: [N_Nodes, N_Nodes]
# H_l+1: [N_Nodes, F]


class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(GCNLayer, self).__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):

        # adj_matrix: Supports directed edges by non-symmetric matrices. 横: 出边
        # Assumes to already have added the identity connections.
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats


#
node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
adj_matrix = torch.Tensor(
    [[[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]])

print("Node features:\n", node_feats)
print("\nAdjacency matrix:\n", adj_matrix)

layer = GCNLayer(c_in=2, c_out=2)
layer.projection.weight.data = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
layer.projection.bias.data = torch.Tensor([0.0, 0.0])


with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)


# [Graph Attention]
# hi, hj: [F]
# W: [F, F]
# a: [1, 2F]

# hi, hj: [I]
# W: [F, H*O]
# a: [H, 1, 2O]

print()


class GATLayer(nn.Module):
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        # c_out=HO
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        #
        self.projection = nn.Linear(c_in, num_heads * c_out)
        self.a = nn.Parameter(torch.Tensor(
            num_heads, 2 * c_out))  # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        #
        nn.init.xavier_uniform_(self.projection.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        # [N, NN, I]; [N, NN, NN]
        batch_size, num_nodes = node_feats.shape[:2]

        #
        node_feats = self.projection(node_feats)  # [N, NN, H * O]
        node_feats = node_feats.view(
            batch_size, num_nodes, self.num_heads, -1)  # [N, NN, H, O]

        # Returns indices where the adjacency matrix is not 0 => edges
        edges = adj_matrix.nonzero()  # as_tuple=False. [X, _]
        node_feats_flat = node_feats.view(
            batch_size * num_nodes, self.num_heads, -1)  # [N*NN, H, O]
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]  # [X]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]  # [X]
        a_input = torch.cat(  # [X, H, 2O]
            [
                node_feats_flat[edge_indices_row],
                node_feats_flat[edge_indices_col],
            ], dim=-1)

        #
        attn_logits = torch.einsum("bhc,hc->bh", a_input, self.a)  # [X, H]
        attn_logits = self.leakyrelu(attn_logits)
        # attn_matrix: [N, NN, NN, H]
        attn_matrix = attn_logits.new_zeros(
            adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(
            1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        #
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            attn_probs_p = attn_probs.permute(0, 3, 1, 2)
            print("Attention probs\n", attn_probs_p)
        node_feats = torch.einsum(
            "bijh,bjhc->bihc", attn_probs, node_feats)  # [N, NN, H, O]

        #
        if self.concat_heads:
            node_feats = node_feats.reshape(
                batch_size, num_nodes, -1)  # [N, NN, HO]
        else:
            node_feats = node_feats.mean(dim=2)  # [N, NN, O]

        return node_feats


# I=2, O=1, H=2
layer = GATLayer(2, 2, num_heads=2)
layer.projection.weight.data = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
layer.projection.bias.data = torch.Tensor([0.0, 0.0])
layer.a.data = torch.Tensor([[-0.2, 0.3], [0.1, -0.1]])


with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix, print_attn_probs=True)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)


# [PyTorch Geometric]

gnn_layer_by_name = {"GCN": geom_nn.GCNConv,
                     "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

# [Experiments on graph structures]


# [Node-level tasks: Semi-supervised node classification]
cora_dataset = geom_datasets.Planetoid(
    root=DATASET_PATH, name="Cora")
print(cora_dataset[0])


class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=2,
        layer_name="GCN",
        dp_rate=0.1,
        **kwargs,
    ):
        # num_layers: Number of "hidden" graph layers (不含输入层, 含输出层)
        # kwargs: (e.g. number of heads for GAT)
        super(GNNModel, self).__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):

        super(MLPModel, self).__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_channels, out_channels),
                       nn.ReLU(inplace=True), nn.Dropout(dp_rate)]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):  # 为了兼容
        return self.layers(x)


class NodeLevelGNN(pl.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super(NodeLevelGNN, self).__init__()
        self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unknown forward mode: %s" % mode

        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]
               ).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)


def train_node_classifier(model_name, dataset, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = geom_loader.DataLoader(dataset, batch_size=1)

    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(
            save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=AVAIL_GPUS,
        max_epochs=200,
        # progress_bar_refresh_rate=0,
        enable_progress_bar=False
    )  # 0 because epoch size is 1
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = False

    pretrained_filename = os.path.join(
        CHECKPOINT_PATH, "NodeLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = NodeLevelGNN(
            model_name=model_name, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs
        )
        trainer.fit(model, node_data_loader, node_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)

    test_result = trainer.test(
        model, dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc, "val": val_acc,
              "test": test_result[0]["test_acc"]}
    return model, result


def print_results(result_dict):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100. * result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100. * result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100. * result_dict["test"]))


node_mlp_model, node_mlp_result = train_node_classifier(
    model_name="MLP", dataset=cora_dataset, c_hidden=16, num_layers=2, dp_rate=0.1
)

print_results(node_mlp_result)

node_gnn_model, node_gnn_result = train_node_classifier(
    model_name="GNN", layer_name="GCN", dataset=cora_dataset, c_hidden=16, num_layers=2, dp_rate=0.1
)
print_results(node_gnn_result)


# [Edge-level tasks: Link prediction]


# [Graph-level tasks: Graph classification]

tu_dataset = geom_datasets.TUDataset(root=DATASET_PATH, name="MUTAG")

print("Data object:", tu_dataset.data)
print("Length:", len(tu_dataset))
print("Average label: %4.2f" % (tu_dataset.data.y.float().mean().item()))
"""
Data object: Data(x=[3371, 7], edge_index=[2, 7442], edge_attr=[7442, 4], y=[188])
Length: 188
Average label: 0.66
"""
torch.manual_seed(42)
tu_dataset.shuffle()
train_dataset = tu_dataset[:150]
test_dataset = tu_dataset[150:]

graph_train_loader = geom_loader.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)

graph_val_loader = geom_loader.DataLoader(test_dataset, batch_size=BATCH_SIZE)
graph_test_loader = geom_loader.DataLoader(test_dataset, batch_size=BATCH_SIZE)


batch = next(iter(graph_test_loader))
print("Batch:", batch)
print("Labels:", batch.y[:10])
print("Batch indices:", batch.batch[:40])


class GraphGNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        # dp_rate_linear: (usually much higher than inside the GNN)
        super(GraphGNNModel, self).__init__()
        self.GNN = GNNModel(c_in=c_in, c_hidden=c_hidden,
                            c_out=c_hidden, **kwargs)  # num_layers, layer_name, dp_rate
        self.head = nn.Sequential(nn.Dropout(
            dp_rate_linear), nn.Linear(c_hidden, c_out))

    def forward(self, x, edge_index, batch_idx):
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)
        x = self.head(x)
        return x


class GraphLevelGNN(pl.LightningModule):
    # {'c_in': 7, 'c_out': 1, 'c_hidden': 256, 'layer_name': 'GraphConv', 'num_layers': 3, 'dp_rate_linear': 0.5, 'dp_rate': 0.0}
    def __init__(self, **model_kwargs):
        super(GraphLevelGNN, self).__init__()
        self.save_hyperparameters()

        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss(
        ) if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)  # data.y: Long
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.0)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)


def train_graph_classifier(model_name, **model_kwargs):
    pl.seed_everything(42)

    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(
            save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=AVAIL_GPUS,
        max_epochs=500,
        enable_progress_bar=False,
    )
    trainer.logger._default_hp_metric = False

    pretrained_filename = os.path.join(
        CHECKPOINT_PATH, "GraphLevel%s.ckpt" % model_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = GraphLevelGNN(
            c_in=tu_dataset.num_node_features,
            c_out=1 if tu_dataset.num_classes == 2 else tu_dataset.num_classes,
            **model_kwargs,
        )
        trainer.fit(model, graph_train_loader, graph_val_loader)
        model = GraphLevelGNN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)

    train_result = trainer.test(
        model, dataloaders=graph_train_loader, verbose=False)
    test_result = trainer.test(
        model, dataloaders=graph_test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"],
              "train": train_result[0]["test_acc"]}
    return model, result


model, result = train_graph_classifier(
    model_name="GraphConv", c_hidden=256, layer_name="GraphConv", num_layers=3, dp_rate_linear=0.5, dp_rate=0.0
)

print("Train performance: %4.2f%%" % (100. * result["train"]))
print("Test performance:  %4.2f%%" % (100. * result["test"]))


# [Conclusion]
