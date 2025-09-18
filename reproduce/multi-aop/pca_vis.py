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

# Fix font issue - use a default font that should be available
plt.rcParams['font.family'] = 'DejaVu Sans'

def feature_generator(model_path, csv_path, seq_faa_path, pooled_seq_path, graph_fea_path, fused_fea_path, last_hidden_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CombinedModel()
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_loader = get_data_loader(csv_path, batch_size=batch_size, seq_length=seq_length)
    test_pred_list = []
    test_target_list = []
    seq_fea_list, pooled_seq_list, graph_fea_list, fused_fea_list, last_hidden_list = [], [], [], [], []
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
            test_pred_list.extend(preds)
            test_target_list.extend(labels.cpu().numpy())

            # Store features
            seq_fea_list.extend(seq_features.cpu().numpy())
            pooled_seq_list.extend(pooled_seq.cpu().numpy())
            graph_fea_list.extend(graph_features.cpu().numpy())
            fused_fea_list.extend(fused_features.cpu().numpy())
            last_hidden_list.extend(last_hidden.cpu().numpy())

    # Calculate accuracy
    test_acc = accuracy_score(test_target_list, test_pred_list)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save features
    np.save(seq_faa_path, np.array(seq_fea_list))
    np.save(pooled_seq_path, np.array(pooled_seq_list))
    np.save(graph_fea_path, np.array(graph_fea_list))
    np.save(fused_fea_path, np.array(fused_fea_list))
    np.save(last_hidden_path, np.array(last_hidden_list))

    return test_acc, np.array(seq_fea_list), np.array(pooled_seq_list), np.array(graph_fea_list), np.array(fused_fea_list), np.array(last_hidden_list)


def pca_ana(csv_path, feature_path, title_text, fig_path):
    seq = pd.read_csv(csv_path)
    features = np.load(feature_path)

    # PCA
    pca = PCA(n_components = 2)
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    components = pipe.fit_transform(features)
    data_pca = pd.DataFrame(components, columns = ['PC1', 'PC2'])
    data_pca.insert(data_pca.shape[1], "labels", seq['label'])
    # Plot PCA
    plt.rcParams['font.family'] = 'sans Serif'
    plt.rcParams['font.serif'] = ['Arial']
    plt.scatter(data_pca.values[:, 0], data_pca.values[:, 1], c=data_pca.values[:, 2])

    # Print variance explained
    explained_variance = pca.explained_variance_ratio_
    print(f"Variance explained by first two components: {explained_variance[0]:.2%}, {explained_variance[1]:.2%}")
    print(f"Total variance explained by all components: {np.sum(explained_variance):.2%}")

    # Add labels
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)', fontsize=14)
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)', fontsize=14)
    plt.title(title_text, size=18, fontweight="bold")
    plt.legend()
    plt.savefig(fig_path)
    return data_pca


if __name__ == '__main__':
    # File paths
    # model_path = "best_model_221316_20250421_AnOxPP/best_model.pth"
    # csv_path = "/home/jianxiu/OneDrive/aop/data/AnOxPP/AnOxPP.csv"
    #
    # seq_faa_path = "/home/jianxiu/OneDrive/aop/data/AnOxPP/AnOxPP_seq.npy"
    # pooled_seq_path = "/home/jianxiu/OneDrive/aop/data/AnOxPP/AnOxPP_pooled_seq.npy"
    # graph_fea_path = "/home/jianxiu/OneDrive/aop/data/AnOxPP/AnOxPP_graph.npy"
    # fused_fea_path = "/home/jianxiu/OneDrive/aop/data/AnOxPP/AnOxPP_fused.npy"
    # last_hidden_path = "/home/jianxiu/OneDrive/aop/data/AnOxPP/AnOxPP_last_hidden.npy"
    #
    # feature_generator(model_path, csv_path, seq_faa_path, pooled_seq_path, graph_fea_path, fused_fea_path, last_hidden_path)
    #
    # # Output paths
    # title_text = "AnOxPP Dataset PCA Visualization"
    # fig_path = "AnOxPP_PCA_last_hidden.svg"
    # pca_ana(csv_path, last_hidden_path, title_text, fig_path)

    model_path = "best_model_152415_AOPP/best_model.pth"
    csv_path = "/home/jianxiu/OneDrive/aop/data/AOPP/AOPP.csv"

    seq_faa_path = "/home/jianxiu/OneDrive/aop/data/AOPP/AOPP_seq.npy"
    pooled_seq_path = "/home/jianxiu/OneDrive/aop/data/AOPP/AOPP_pooled_seq.npy"
    graph_fea_path = "/home/jianxiu/OneDrive/aop/data/AOPP/AOPP_graph.npy"
    fused_fea_path = "/home/jianxiu/OneDrive/aop/data/AOPP/AOPP_fused.npy"
    last_hidden_path = "/home/jianxiu/OneDrive/aop/data/AOPP/AOPP_last_hidden.npy"

    test_acc, seq_fea_list, pooled_seq_list, graph_fea_list, fused_fea_list, last_hidden_list = feature_generator(model_path, csv_path, seq_faa_path, pooled_seq_path, graph_fea_path, fused_fea_path, last_hidden_path)

    # Output paths
    title_text = "AOPP Dataset PCA Visualization"
    fig_path = "AOPP_PCA_last_hidden.svg"
    pca_ana(csv_path, last_hidden_path, title_text, fig_path)
    print('smart')