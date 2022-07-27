import streamlit as st


st.title('GCN Architectures used in this tutorial')


col1, col2 = st.columns(2)

with col1:
    col1.header('2 Layer GCN')
    col1.image('figures/2layer_gcn.png')

with col2:
    col2.header('Random GCN')
    col2.image('figures/random_gcn.png')
