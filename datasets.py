import torch_geometric.datasets as tgd
import torch_geometric.transforms as tgt


def load_karate_club():
    data = tgd.KarateClub()[0]

    return data


def load_cora():
    data = tgd.Planetoid(root='./data', name='Cora', transform=tgt.NormalizeFeatures())[0]

    return data
