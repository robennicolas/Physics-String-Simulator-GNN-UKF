import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class PhysicStringGNN(MessagePassing):
    def __init__(self, node_features, edge_features, hidden_dim, num_layers=3):
        super().__init__(aggr='add')
        self.num_layers = num_layers

        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.mlp2 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.decoder = nn.Linear(hidden_dim, 2)


    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # On fait circuler l'information plusieurs fois
        for _ in range(self.num_layers):
            # On propage et on met à jour x à chaque étape
            x = x + self.propagate(edge_index, x=x, edge_attr=edge_attr)
            
        return self.decoder(x)
    
    def message(self, x_i, x_j, edge_attr):
        concat = torch.cat([x_i, x_j, edge_attr], dim=-1)  # 3*hidden_dim
        return self.mlp1(concat)


    def update(self, aggr_out, x):
        concat = torch.cat([x, aggr_out], dim=-1)  # 2*hidden_dim
        return self.mlp2(concat)
