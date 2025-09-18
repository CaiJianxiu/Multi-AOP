import sys
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from aop_dataloader import *
from seq_model_def import *
from graph_model_def import *
from aop_def import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

# Define global constants
seq_length = 50  # Add this from the training script
batch_size = 500


def feature_generator(model_path, csv_path, seq_feature_path, graph_feature_path, fused_feature_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CombinedModel()
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loading data from: {csv_path}")
    test_loader = get_data_loader(csv_path, batch_size=batch_size, seq_length=seq_length)

    test_pred_list = []
    test_target_list = []
    seq_features = []
    graph_features = []
    fused_features = []
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
            # Forward pass - get all feature types
            seq_fea, graph_fea, fused_fea, outputs = model(sequences, x, edge_index, edge_attr, batch_idx_tensor)

            # Track predictions
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(float)
            test_pred_list.extend(preds)
            test_target_list.extend(labels.cpu().numpy())

            # Store all feature types separately
            seq_features.extend(seq_fea.cpu().numpy())
            graph_features.extend(graph_fea.cpu().numpy())
            fused_features.extend(fused_fea.cpu().numpy())
    # Calculate accuracy
    test_acc = accuracy_score(test_target_list, test_pred_list)
    print(f"Test accuracy: {test_acc:.4f}")
    # Save features
    print(f"Saving features to:")
    print(f"  - Sequence: {seq_feature_path}")
    print(f"  - Graph: {graph_feature_path}")
    print(f"  - Fused: {fused_feature_path}")
    np.save(seq_feature_path, seq_features)
    np.save(graph_feature_path, graph_features)
    np.save(fused_feature_path, fused_features)
    return test_acc

def tsne_ana(csv_path, feature_path, title_text, fig_path, feature_type):
    print(f"Performing t-SNE analysis for {feature_type} features...")

    # Load data
    seq = pd.read_csv(csv_path)
    features = np.load(feature_path)
    labels = seq['label'].values

    # Print feature shape information
    print(f"Feature shape: {features.shape}")
    print(f"Number of samples: {len(labels)}")

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                n_iter=1000, learning_rate=200)
    print("Fitting t-SNE (this may take a while)...")
    components = tsne.fit_transform(features)

    # Min-max scaling for visualization
    x_min, x_max = np.min(components, 0), np.max(components, 0)
    tsne_features = (components - x_min) / (x_max - x_min)

    # Create DataFrame for easier plotting
    df = pd.DataFrame()
    df["comp1"] = tsne_features[:, 0]
    df["comp2"] = tsne_features[:, 1]
    df["label"] = labels

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    # Create scatter plot with color based on labels
    scatter = plt.scatter(df["comp1"], df["comp2"],
                          s=10, c=df["label"], marker='.', linewidths=0.2,
                          cmap='viridis', alpha=0.8)
    cbar = plt.colorbar(scatter, label='Label')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0', '1'])
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    full_title = f"{title_text} - {feature_type} Features"
    plt.title(full_title, size=18, fontweight="bold")

    plt.tight_layout()
    print(f"Saving t-SNE plot to: {fig_path}")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # File paths
    model_path = "best_model_221316_20250421_AnOxPP/best_model.pth"
    csv_path = "/home/jianxiu/OneDrive/aop/data/AnOxPP/AnOxPP.csv"
    seq_feature_path = "/home/jianxiu/OneDrive/aop/data/AnOxPP/AnOxPP_seq.npy"
    graph_feature_path = "/home/jianxiu/OneDrive/aop/data/AnOxPP/AnOxPP_graph.npy"
    fused_feature_path = "/home/jianxiu/OneDrive/aop/data/AnOxPP/AnOxPP_fused.npy"
    # Output paths
    title_text = "AnOxPP dataset"
    seq_fig_path = "AnOxPP_tsne_seq.svg"
    graph_fig_path = "AnOxPP_tsne_graph.svg"
    fused_fig_path = "AnOxPP_tsne_fused.svg"
    print("Starting feature extraction and t-SNE analysis pipeline...")

    test_acc = feature_generator(model_path, csv_path,
                                 seq_feature_path,
                                 graph_feature_path,
                                 fused_feature_path)
    # Perform t-SNE analysis for each feature type
    # tsne_ana(csv_path, seq_feature_path, title_text, seq_fig_path, "Sequence")
    # tsne_ana(csv_path, graph_feature_path, title_text, graph_fig_path, "Graph")
    tsne_ana(csv_path, fused_feature_path, title_text, fused_fig_path, "Fused")

    print("smart")