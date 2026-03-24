import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
from pathlib import Path

class PhysicStringDataset(Dataset):
    """
    Each sample is a tuple (graph_t, graph_t+1).
    Input  : state at t   → features nodes + edges
    Target : acceleration at t+1 
    """

    def __init__(self, data_dir="../data"):
        super().__init__()
        data_dir = Path(data_dir)

        self.nodes_df = pd.read_csv(data_dir / "nodes.csv")
        self.edges_df = pd.read_csv(data_dir / "edges.csv")

        node_cols = ["x", "y", "vx", "vy", "ax", "ay"]

        self.node_mean = self.nodes_df[node_cols].mean()
        self.node_std  = self.nodes_df[node_cols].std()

        self.nodes_df[node_cols] = (self.nodes_df[node_cols] - self.node_mean) / self.node_std


        edge_cols = ["length", "tension", "dir_x", "dir_y"]

        self.edge_mean = self.edges_df[edge_cols].mean()
        self.edge_std  = self.edges_df[edge_cols].std()

        self.edges_df[edge_cols] = (self.edges_df[edge_cols] - self.edge_mean) / self.edge_std


        print("// --- COPY PASTE IN THE PHYSICSTRING C++ ---")
        print("// Normalisation constantes in Dataset Python")

        for col in ["x", "y", "vx", "vy"]:
            print(f"const float MEAN_{col.upper()} = {self.node_mean[col]:.6f}f;")
            print(f"const float STD_{col.upper()}  = {self.node_std[col]:.6f}f;")

        for col in ["ax", "ay"]: # Pour la dénormalisation de la sortie !
            print(f"const float MEAN_{col.upper()} = {self.node_mean[col]:.6f}f;")
            print(f"const float STD_{col.upper()}  = {self.node_std[col]:.6f}f;")

        for col in ["length", "tension", "dir_x", "dir_y"]:
            print(f"const float MEAN_{col.upper()} = {self.edge_mean[col]:.6f}f;")
            print(f"const float STD_{col.upper()}  = {self.edge_std[col]:.6f}f;")
        print("// ------------------------------------")

        frames = (
            self.nodes_df[["snapshot_id", "frame_id"]]
            .drop_duplicates()
            .sort_values(["snapshot_id", "frame_id"])
            .values.tolist()
        )

        # We build pairs (t, t+1) inside the same snapshot
        self.pairs = []
        for i in range(len(frames) - 1):
            snap_t,  frame_t  = frames[i]
            snap_t1, frame_t1 = frames[i + 1]
            if snap_t == snap_t1:   # same snapshot
                self.pairs.append((snap_t, frame_t, frame_t1))

    def len(self):
        return len(self.pairs)

    def get(self, idx):
        snap, frame_t, frame_t1 = self.pairs[idx]

        graph_t  = self._build_graph(snap, frame_t)
        graph_t1 = self._build_graph(snap, frame_t1)

        # Target = graph acc at t+1
        graph_t.y = graph_t1.y.clone() 

        return graph_t

    def _build_graph(self, snapshot_id, frame_id):
        # ---- Nodes ----
        n = self.nodes_df[
            (self.nodes_df["snapshot_id"] == snapshot_id) &
            (self.nodes_df["frame_id"]    == frame_id)
        ].sort_values("node_id")

        # Features nodes : [x, y, vx, vy, is_fixed]  → 5 features  
        # We took off the acceleration because its the target 
        node_features = torch.tensor(
            n[["x", "y", "vx", "vy", "is_fixed"]].values,
            dtype=torch.float
        )

        target_accel = torch.tensor(
            n[["ax", "ay"]].values,
            dtype=torch.float
        )


        # ---- Edges ----
        e = self.edges_df[
            (self.edges_df["snapshot_id"] == snapshot_id) &
            (self.edges_df["frame_id"]    == frame_id)
        ].sort_values("node_i")

        # edge_index : [2, num_edges] — non directed graph
        src = torch.tensor(e["node_i"].values, dtype=torch.long)
        dst = torch.tensor(e["node_j"].values, dtype=torch.long)
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src])
        ], dim=0)

        # Features edges : [length, tension, dir_x, dir_y] → 4 features
        # Duplicated for the two directions
        edge_attr_base = torch.tensor(
            e[["length", "tension", "dir_x", "dir_y"]].values,
            dtype=torch.float
        )
        edge_attr_reverse = edge_attr_base.clone()
        edge_attr_reverse[:, 2:4] *= -1  
        edge_attr = torch.cat([edge_attr_base, edge_attr_reverse], dim=0)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=target_accel)


# --- Test ---
if __name__ == "__main__":
    dataset = PhysicStringDataset(data_dir="../data")
    print(f"Numbre of samples : {len(dataset)}")

    sample = dataset[0]
    print(f"Node features : {sample.x.shape}")    # [n_nodes, 7]
    print(f"Edge index    : {sample.edge_index.shape}")  # [2, 2*(n-1)]
    print(f"Edge attr     : {sample.edge_attr.shape}")   # [2*(n-1), 4]
    print(f"Target (acc)  : {sample.y.shape}")    # [n_nodes, 2]