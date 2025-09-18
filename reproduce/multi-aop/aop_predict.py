import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from transformers.models.aria.modeling_aria import sequential_experts_gemm

from aop_dataloader import *
from seq_model_def import *
from graph_model_def import *
from aop_def import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Define global constants
seq_length = 50
batch_size = 500

def aop_predict(model_path, csv_path, out_csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CombinedModel()
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_df = pd.read_csv(csv_path)
    if 'label' not in test_df.columns:
        test_df['label'] = 0 * len(test_df)
        test_df.to_csv(csv_path, index=False)
    test_loader = get_data_loader(csv_path, batch_size=batch_size, seq_length=seq_length)
    test_prob_list = []
    test_pred_list = []
    test_target_list = []

    print("Extracting features...")
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            sequences = batch['sequences'].to(device)
            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr = batch['edge_attr'].to(device)
            batch_idx_tensor = batch['batch'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            seq_features, pooled_seq, graph_features, fused_features, last_hidden, outputs = model(sequences, x, edge_index, edge_attr, batch_idx_tensor)

            # Track predictions
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(float)
            test_prob_list.append(probs)
            test_pred_list.extend(preds)
            test_target_list.extend(labels.cpu().numpy())
    test_df['probs'] = test_prob_list
    test_df['preds'] = test_pred_list
    test_df.to_csv(out_csv_path, index=False)
    return test_df

if __name__ == "__main__":
    model_path = '/home/jianxiu/OneDrive/aop/external/aop_final_model/best_model.pth'
    csv_path = '/home/jianxiu/OneDrive/aop/fruit/citrus_len50.csv'
    out_csv_path = '/home/jianxiu/OneDrive/aop/fruit/citrus_len50_pred.csv'
    aop_predict(model_path, csv_path, out_csv_path)
    print('smart')

