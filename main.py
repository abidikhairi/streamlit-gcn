import streamlit as st
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets import load_cora, load_karate_club
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from model import GCN

st.set_page_config(layout="wide")


dataset = st.sidebar.selectbox("Select dataset", ['karate club', 'cora'])

if dataset == 'cora':
    data = load_cora()
elif dataset == 'karate club':
    data = load_karate_club()


with st.container():
    col1, col2 = st.columns(2)
    with col1:
        col1.title(f'Graph {dataset}')
        graph = to_networkx(data).to_undirected()

        fig, ax = plt.subplots(figsize=(10, 10))

        nx.draw(graph, ax=ax, node_size=100, node_color=data.y.numpy(), cmap='Paired')

        st.pyplot(fig)
    
    with col2:
        col2.title('GCN Hyper-Parameters')

        nhid = st.slider('Number of hidden units', 8, 64, 1)
        lr = st.number_input('Learning rate', 0.01, 0.1, 0.05)
        epochs = st.number_input('Number of training epochs', 1, 100, 1)

        train = st.button('Train')
        feature_extractor = st.button('Use Randomly Initialized GCN')

if train:
    st.header('Running training')
    nclass = data.y.max().item() + 1
    model = GCN(data.num_features, nhid, nclass)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    model.train()
    pbar = st.progress(0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.edge_index, data.x)
        logits = F.log_softmax(out, dim=1)
        loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])

        loss.backward()        
        optimizer.step()
        pbar.progress(epoch / epochs)

    pbar.text('Training finished')

    model.eval()
    h = model(data.edge_index, data.x).detach().numpy()
    x = TSNE(n_components=2).fit_transform(h)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x[:, 0], x[:, 1], c=data.y.numpy(), cmap='Paired')
    st.pyplot(fig)

if feature_extractor:
    from model import FeatureExtractorGCN
    model = FeatureExtractorGCN(data.num_features)
    model.eval()

    h = model(data.edge_index, data.x).detach().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(h[:, 0], h[:, 1], c=data.y.numpy(), s=20, cmap='Paired')
    st.pyplot(fig)
