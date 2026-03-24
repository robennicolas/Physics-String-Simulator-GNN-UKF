import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from dataset import PhysicStringDataset
from model import PhysicStringGNN



def inject_noise(batch, noise_std=0.02):
    noisy_batch = batch.clone() 

    mask = (noisy_batch.x[:, 4] == 0).unsqueeze(1)

    noise = torch.randn_like(noisy_batch.x[:, :4]) * noise_std

    noisy_batch.x[:, :4] += (noise * mask)
    
    return noisy_batch


if __name__ == "__main__":

    #---- VARIABLES DECLARATIONS-----
    node_features = 5
    edge_features = 4
    hidden_dim = 128
    num_layers = 3
    batch_size = 32
    lr = 1e-3
    num_epochs = 300

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    

    #---- DATASET -----
    dataset = PhysicStringDataset(data_dir="../data")


    # split the dataset into three parts (train 70%, test 15%, validation 15%)
    test_size = 0.15
    val_size = 0.15

    test_amount, val_amount = int(dataset.__len__() * test_size), int(dataset.__len__() * val_size)


    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
                (dataset.__len__() - (test_amount + val_amount)), 
                test_amount, 
                val_amount
    ])


    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    

    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    torch.save(test_set.indices, "test_indices.pt")


    model = PhysicStringGNN(node_features, edge_features, hidden_dim, num_layers)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()



    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_data:
            batch = batch.to(device)
            optimizer.zero_grad() # On remet les gradients à zéro d'abord
            batch = inject_noise(batch, noise_std=0.02)
            
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss_val = loss(pred, batch.y)
            
            loss_val.backward()            
            optimizer.step()               
            total_loss += loss_val.item()
        
        train_loss = total_loss / len(train_data)

        # --- VALIDATION ---
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch in val_data:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr)
                loss_val = loss(pred, batch.y)
                val_loss_total += loss_val.item() # <-- On accumule la loss ici !
        
        val_loss = val_loss_total / len(val_data)
        
        # On affiche les deux pour vérifier qu'il n'y a pas d'overfitting
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        

    torch.save(model.state_dict(), "model.pt")


