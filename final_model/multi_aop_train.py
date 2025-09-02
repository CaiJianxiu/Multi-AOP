import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import os
from tqdm import tqdm
import json
import random
from aop_dataloader import *
from aop_def import *
from datetime import datetime
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import numpy as np


def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def train_and_evaluate(my_seed, device, train_loader, val_loader, output_dir,
                       learning_rate, weight_decay,
                       max_epochs, patience):
    # tmp_seed = random.randint(1, 10000)
    tmp_seed = 221316
    torch.manual_seed(tmp_seed)
    model = CombinedModel().to(device)
    # Initialize optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience, verbose=True
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": []
    }

    # Training loop
    best_val_fnr = 1.0
    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds_train = []
        all_labels_train = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} (Train)")
        for batch in progress_bar:
            # Move data to device
            sequences = batch['sequences'].to(device)
            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr = batch['edge_attr'].to(device)
            batch_idx_tensor = batch['batch'].to(device)
            labels = batch['labels'].to(device)
            # Forward pass
            optimizer.zero_grad()
            seq_features, pooled_seq, graph_features, fused_features, last_hidden, outputs = model(sequences, x, edge_index, edge_attr, batch_idx_tensor)
            # Compute loss
            loss = criterion(outputs.squeeze(), labels)
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            preds = (outputs.squeeze() > 0.5).float()
            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels_train, all_preds_train)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds_val = []
        all_labels_val = []
        all_probs_val = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} (Val)"):
                # Move data to device
                sequences = batch['sequences'].to(device)
                x = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr = batch['edge_attr'].to(device)
                batch_idx_tensor = batch['batch'].to(device)
                labels = batch['labels'].to(device)
                # Forward pass
                seq_features, pooled_seq, graph_features, fused_features, last_hidden, outputs = model(sequences, x, edge_index, edge_attr, batch_idx_tensor)

                # Compute loss
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

                # Track predictions
                probs = outputs.squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(float)
                all_probs_val.extend(probs)
                all_preds_val.extend(preds)
                all_labels_val.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_acc = accuracy_score(all_labels_val, all_preds_val)
        val_loss = val_loss / len(val_loader)
        tn, fp, fn, tp = confusion_matrix(all_labels_val, all_preds_val, labels=[0, 1]).ravel()
        val_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        val_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)

        # Print progress
        print(f"Epoch [{epoch + 1}/{max_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}, False negative rate: {val_fnr:.4f}, True positive rate: {val_tpr:.4f}")

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_metrics': {
                    'acc': val_acc,
                    'fnr': val_fnr,
                    'tpr': val_tpr
                }
            }, os.path.join(output_dir, "best_model.pth"))

            print(f"New best model saved with val_acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Final evaluation on validation set
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create results dictionary
    results = {
        "seed": my_seed,
        "epochs_trained": epoch + 1,
        "best_epoch": checkpoint['epoch'] + 1,
        "val_acc": checkpoint['val_metrics']['acc'],
        "val_fnr": checkpoint['val_metrics']['fnr'],
        "val_tpr": checkpoint['val_metrics']['tpr']
    }
    return model, results


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # file configuration
    train_dir = 'combined_data.csv'
    val_dir = 'external_aop.csv'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size = 16
    train_loader = get_data_loader(train_dir, batch_size, seq_length)
    val_loader = get_data_loader(val_dir, batch_size, seq_length)
    # Load data
    print("Loading data...")

    # result_dict
    repeat_results = {
        "seed": [],
        "epochs_trained": [],
        "best_epoch": [],
        "val_acc": [],
        "val_fnr": [],
        "val_tpr": []
    }
    #
    # tmp_seed = random.randint(1, 10000)
    tmp_seed = 221316
    torch.manual_seed(tmp_seed)
    final_output_dir = f"best_model_{tmp_seed}_{timestamp}"
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Training with best parameters... Results will be saved to: {final_output_dir}")

    model, results = train_and_evaluate(
        my_seed = tmp_seed, device = device,
        train_loader = train_loader, val_loader = val_loader, output_dir = final_output_dir,
        learning_rate = 5e-4, weight_decay = 3e-3, max_epochs = 200, patience = 50
    )
    repeat_results["seed"].append(tmp_seed)
    repeat_results["epochs_trained"].append(results["epochs_trained"])
    repeat_results["best_epoch"].append(results["best_epoch"])
    repeat_results["val_acc"].append(results["val_acc"])
    repeat_results["val_fnr"].append(results["val_fnr"])
    repeat_results["val_tpr"].append(results["val_tpr"])
    result_df = pd.DataFrame(repeat_results)
    result_df.to_csv("external_results.csv")
    print("smart")
