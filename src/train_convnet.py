import torch 
import copy
from torch import nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


def train_model(model, X_train, X_val, y_train, y_val, epochs=50, ckpt_name="cnn_baseline.pth"):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Change to Tensor
    y_train = torch.from_numpy(y_train).float()
    y_val = torch.from_numpy(y_val).float()

    # Record
    data = {}

    # Create loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Set batches
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())

    # Building training and evaluating loop
    for epoch in range(epochs):
        ### Training 
        model.train()
        running_loss, correct, total = 0.0, 0.0, 0.0
        tp, fp, fn = 0.0, 0.0, 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            flip = torch.rand(X_batch.size(0), device=device) < 0.5
            X_batch = torch.where(flip[:, None, None, None], X_batch.flip(3), X_batch)

            # Forward pass
            y_logits = model(X_batch) # model outputs raw logits

            # Calculate loss
            loss = loss_fn(y_logits, y_batch)

            # Optimizer zero grad
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            
            preds = (y_logits > 0).float()

            # Accuracy
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()

            # F1
            tp += ((preds == 1) & (y_batch == 1)).sum().item()
            fp += ((preds == 1) & (y_batch == 0)).sum().item()
            fn += ((preds == 0) & (y_batch == 1)).sum().item()




        train_loss = running_loss / n_batches
        train_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        train_accuracy = correct / total


        val_running_loss, val_correct, val_total = 0.0, 0.0, 0.0
        val_tp, val_fp, val_fn = 0.0, 0.0, 0.0
        val_n_batches = 0

        ### Testing
        model.eval()
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            with torch.inference_mode():
                # Forward pass
                val_logits = model(X_batch)

                loss = loss_fn(val_logits, y_batch)

                val_running_loss += loss.item()
                val_n_batches += 1
                
                            
                preds = (val_logits > 0).float()
                
                # Accuracy
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.numel()
                
                # F1
                val_tp += ((preds == 1) & (y_batch == 1)).sum().item()
                val_fp += ((preds == 1) & (y_batch == 0)).sum().item()
                val_fn += ((preds == 0) & (y_batch == 1)).sum().item()

        val_loss = val_running_loss / val_n_batches
        val_f1 = 2 * val_tp / max(2 * val_tp + val_fp + val_fn, 1)
        val_accuracy = val_correct / val_total

        if best_acc < val_accuracy:
            best_acc = val_accuracy 
            best_model = copy.deepcopy(model.state_dict())
            
        data[epoch] = {"train_loss": train_loss, "train_f1": train_f1,
                       "train_acc": train_accuracy, "val_loss": val_loss, "val_f1": val_f1, "val_acc": val_accuracy}
       

        # Print results
        print(f"Epoch: {epoch} | Loss: {train_loss:.4f} | F1: {train_f1:.4f} | Acc: {train_accuracy:.4f}| Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_accuracy:.4f}")



    # Pass the best model
    model.load_state_dict(best_model)

    # Saving our model as_model.pth
    MODEL_PATH = Path("checkpoints")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = ckpt_name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=best_model, f=MODEL_SAVE_PATH)

    return data

def eval_cnn_model(model, X_test, y_test, ckpt_name="cnn_baseline.pth"):
    
    # Change to Tensor
    y_test = torch.from_numpy(y_test).float()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.load_state_dict(torch.load("checkpoints/cnn_baseline.pth", map_location=device, weights_only=True))
    loss_fn = nn.BCEWithLogitsLoss()

    # Set test batches
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    running_loss, correct, total = 0.0, 0.0, 0.0
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    n_batches = 0

    model.eval()

    for X_batch, y_batch in test_loader:
        # Send batches to device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        with torch.inference_mode():
            y_logits = model(X_batch)
            preds = (y_logits > 0).float()

            # Loss
            loss = loss_fn(y_logits, y_batch)
            running_loss += loss.item()
            n_batches += 1
            
            # Accuracy
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()
            
            # F1
            tp += ((preds == 1) & (y_batch == 1)).sum().item()
            tn += ((preds == 0) & (y_batch == 0)).sum().item()
            fp += ((preds == 1) & (y_batch == 0)).sum().item()
            fn += ((preds == 0) & (y_batch == 1)).sum().item()

    test_loss = running_loss / n_batches
    f1_dog = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_cat = 2 * tn / max(2 * tn + fn + fp, 1)
    f1 = (f1_dog + f1_cat) / 2
    test_accuracy = correct / total


    data = {"test_loss": test_loss , "f1_dog": f1_dog, "f1_cat": f1_cat, "test_f1": f1, "test_acc": test_accuracy}

    return data