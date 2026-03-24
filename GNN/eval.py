import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from dataset import PhysicStringDataset
from model import PhysicStringGNN

if __name__ == "__main__":

    # ---- VARIABLES ----
    node_features = 5
    edge_features = 4
    hidden_dim    = 128
    num_layers    = 3
    batch_size    = 32

    # CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cpu")
    torch.backends.cudnn.benchmark = True

    # ---- MODELE ----
    model = PhysicStringGNN(node_features, edge_features, hidden_dim, num_layers)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model = model.cpu()  # forced CPU
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save("model_scripted.pt")

    # ---- DATASET ----
    dataset     = PhysicStringDataset(data_dir="../data")
    test_indices = torch.load("test_indices.pt")
    test_set    = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    loss_fn = nn.MSELoss()

    # ---- ONE-STEP EVALUATION ----

    total_loss = 0
    all_preds = []
    all_targets = []

    node_to_track = 2 
    tracked_preds = []
    tracked_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            

            current_loss = loss_fn(pred, batch.y)
            total_loss += current_loss.item()


            p = pred.cpu()
            t = batch.y.cpu()

            for i in range(len(batch.ptr) - 1):
                start = batch.ptr[i]
                tracked_preds.append(p[start + node_to_track])
                tracked_targets.append(t[start + node_to_track])

    print(f"Test MSE Loss : {total_loss / len(test_loader):.6f}")

    # ---- VISUALISATION ----

    preds_plot = torch.stack(tracked_preds).numpy()
    targets_plot = torch.stack(tracked_targets).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8)) 

    # Accélération X
    axes[0].plot(targets_plot[:200, 0], 'g-', label="Réel (Physique)", alpha=0.6, linewidth=2)
    axes[0].plot(preds_plot[:200, 0], 'r--', label="Prédit (GNN)", alpha=0.8)
    axes[0].set_title(f"Évolution de Ax pour le nœud {node_to_track}")
    axes[0].legend()
    axes[0].grid(True)

    # Accélération Y
    axes[1].plot(targets_plot[:200, 1], 'g-', label="Réel (Physique)", alpha=0.6, linewidth=2)
    axes[1].plot(preds_plot[:200, 1], 'r--', label="Prédit (GNN)", alpha=0.8)
    axes[1].set_title(f"Évolution de Ay pour le nœud {node_to_track}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()