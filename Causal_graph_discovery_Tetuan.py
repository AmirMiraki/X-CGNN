import os
os.environ['CASTLE_BACKEND'] = 'pytorch'

from castle.common import GraphDAG
from castle.algorithms import NotearsNonlinear
import networkx as nx
from IPython.display import Image, display
import pandas as pd
import numpy as np
X = pd.read_csv('Data/Tetuan/Tetuan City power consumption.csv',index_col='DateTime')
print(X.head())
X1 = X.drop(columns=['Zone 2  Power Consumption' , 'Zone 3  Power Consumption']).to_numpy()
# notears-nonlinear learn
nt1 = NotearsNonlinear(device_type='gpu')
nt1.learn(X1[:36,:])

X2 = X.drop(columns=['Zone 1 Power Consumption' , 'Zone 3  Power Consumption']).to_numpy()
# notears-nonlinear learn
nt2 = NotearsNonlinear(device_type='gpu')
nt2.learn(X2[:36,:])

X3 = X.drop(columns=['Zone 2  Power Consumption' , 'Zone 1 Power Consumption']).to_numpy()
# notears-nonlinear learn
nt3 = NotearsNonlinear(device_type='gpu')
nt3.learn(X3[:36,:])

# plot
GraphDAG(nt1.causal_matrix)
GraphDAG(nt2.causal_matrix)
GraphDAG(nt3.causal_matrix)

G = nx.DiGraph()
for i in range(len(nt1.causal_matrix)):
  for j in range(len(nt1.causal_matrix)):
    if nt1.causal_matrix[i,j]==1:
      G.add_edge(X.columns[i],X.columns[j])

nx.draw(G,with_labels=True)
pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
plt = Image(pydot_graph.create_png())
display(plt)

G = nx.DiGraph()
for i in range(len(nt2.causal_matrix)):
  for j in range(len(nt2.causal_matrix)):
    if nt2.causal_matrix[i,j]==1:
      G.add_edge(X.columns[i],X.columns[j])

nx.draw(G,with_labels=True)
pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
plt = Image(pydot_graph.create_png())
display(plt)

G = nx.DiGraph()
for i in range(len(nt3.causal_matrix)):
  for j in range(len(nt3.causal_matrix)):
    if nt3.causal_matrix[i,j]==1:
      G.add_edge(X.columns[i],X.columns[j])

nx.draw(G,with_labels=True)
pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
plt = Image(pydot_graph.create_png())
display(plt)

adj = nt1.causal_matrix
adj = np.append(adj,[adj[-1,:],adj[-1,:]],axis=0)
adj = np.append(adj.T,[adj[:,-1],adj[:,-1]],axis=0).T
np.save('Data/Tetuan/causal_adj_Bz1.npy',adj)


W = nt1.weight_causal_matrix
W = np.append(W,[W[-1,:],W[-1,:]],axis=0)
W = np.append(W.T,[W[:,-1],W[:,-1]],axis=0).T
np.save('Data/Tetuan/Tetuan_graph.npy',W)

G = nx.DiGraph()
for i in range(len(adj)):
  for j in range(len(adj)):
    if adj[i,j]==1:
      G.add_edge(X.columns[i],X.columns[j])

nx.draw(G,with_labels=True)
pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
plt = Image(pydot_graph.create_png())
display(plt)