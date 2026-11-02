import torch 
import copy
from torch import nn
from pathlib import Path
from src.metrics import dice, iou
from torch.utils.data import TensorDataset, DataLoader


def train_model(model, X_train, X_val, y_train, y_val, epochs=50):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Record
    data = {}

    # Create loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Set batches
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_loss = float("inf")
    best_model = copy.deepcopy(model.state_dict())

    # Building training and evaluating loop
    for epoch in range(epochs):
        ### Training 
        model.train()
        running_loss, running_dice, running_iou, n_batches = 0.0, 0.0, 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_logits = model(X_batch) # model outputs raw logits
            y_preds = (torch.sigmoid(y_logits) > 0.5).float()

            # Calculate loss
            loss = loss_fn(y_logits, y_batch)

            # Optimizer zero grad
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            running_dice +=  dice(y_preds, y_batch).item()
            running_iou += iou(y_preds, y_batch).item()
            n_batches += 1

        train_loss = running_loss / n_batches
        train_dice = running_dice / n_batches
        train_iou = running_iou / n_batches

        val_running_loss, val_running_dice, val_running_iou, val_n_batches = 0.0, 0.0, 0.0, 0

        ### Testing
        model.eval()
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            with torch.inference_mode():
                # Forward pass
                val_logits = model(X_batch)
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()

                # Calculate val loss and val dice
                val_running_loss += loss_fn(val_logits, y_batch).item()
                val_running_dice += dice(val_preds, y_batch).item()
                val_running_iou += iou(val_preds, y_batch).item()
                val_n_batches += 1

        val_loss = val_running_loss / val_n_batches
        val_dice = val_running_dice / val_n_batches
        val_iou = val_running_iou / val_n_batches

        if best_loss > val_loss:
            best_loss = val_loss 
            best_model = copy.deepcopy(model.state_dict())
            
        data[epoch] = {"train_loss": train_loss, "train_dice": train_dice,
                       "train_iou": train_iou, "val_loss": val_loss, "val_dice": val_dice, "val_iou": val_iou}
       

        # Print results
        print(f"Epoch: {epoch} | Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}| Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")



    # Pass the best model
    model.load_state_dict(best_model)

    # Saving our model as_model.pth
    MODEL_PATH = Path("checkpoints")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "unet_seg.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=best_model, f=MODEL_SAVE_PATH)

    return data

def eval_seg_model(model, X_test, y_test):

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.load_state_dict(torch.load("checkpoints/unet_seg.pth", map_location=device, weights_only=True))
    loss_fn = nn.BCEWithLogitsLoss()

    # Set test batches
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_running_dice, test_running_iou, test_running_loss, test_n_batches = 0.0, 0.0, 0.0, 0

    model.eval()

    for X_batch, y_batch in test_loader:
        # Send batches to device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        with torch.inference_mode():
            y_logits = model(X_batch)
            y_preds = (torch.sigmoid(y_logits) > 0.5).float()

            test_running_loss += loss_fn(y_logits, y_batch).item()
            test_running_iou += iou(y_preds, y_batch).item()
            test_running_dice += dice(y_preds, y_batch).item()
            test_n_batches += 1
    data = {"test_loss": test_running_loss / test_n_batches , "test_dice": test_running_dice / test_n_batches,
            "test_iou": test_running_iou / test_n_batches}

    return data

