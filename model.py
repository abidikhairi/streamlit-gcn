import torch as th
import torch.nn as nn
import torch_geometric.nn as tgnn


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass) -> None:
        super().__init__()

        self.conv1 = tgnn.GraphConv(nfeat, nhid)
        self.conv2 = tgnn.GraphConv(nhid, nclass)

    def forward(self, edge_index, x):
        x = th.relu(self.conv1(x, edge_index))
        x = th.dropout(x, p=0.5, train=self.training)
        x = self.conv2(x, edge_index)

        return x


class FeatureExtractorGCN(nn.Module):
    def __init__(self, nfeats) -> None:
        super().__init__()

        self.conv1 = tgnn.GraphConv(nfeats, 4)
        self.conv2 = tgnn.GraphConv(4, 4)
        self.conv3 = tgnn.GraphConv(4, 2)

    def forward(self, edge_index, x):
        x = th.tanh(self.conv1(x, edge_index))
        x = th.tanh(self.conv2(x, edge_index))
        x = th.tanh(self.conv3(x, edge_index))

        return x
