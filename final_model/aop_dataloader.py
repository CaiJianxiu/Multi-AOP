import torch
from torch.utils.data import Dataset, DataLoader
import rdkit
from rdkit import Chem
from torch_geometric.data import Data, Dataset as GeometricDataset
from torch_geometric.loader import DataLoader as GraphDataLoader
import pandas as pd

# data processing
def aa_to_int(sequence):
    aa_to_int_dict = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }
    return [aa_to_int_dict.get(aa.upper(), -1) for aa in sequence]

def aa_to_smiles(sequence):
    aa_to_smiles_dict = {
        'A': 'CC(N)C(=O)O', 'R': 'NC(=N)NCCCC(N)C(=O)O', 'N': 'NC(=O)CC(N)C(=O)O',
        'D': 'OC(=O)CC(N)C(=O)O', 'C': 'SC(C(N)C(=O)O)', 'E': 'OC(=O)CCC(N)C(=O)O',
        'Q': 'NC(=O)CCC(N)C(=O)O', 'G': 'NCC(=O)O', 'H': 'NC(Cc1c[nH]cn1)C(=O)O',
        'I': 'CC(C)CC(N)C(=O)O', 'L': 'CC(C)CC(N)C(=O)O', 'K': 'NCCCCC(N)C(=O)O',
        'M': 'CSCCC(N)C(=O)O', 'F': 'NC(Cc1ccccc1)C(=O)O', 'P': 'O=C(O)C1CCCN1',
        'S': 'OCC(N)C(=O)O', 'T': 'CC(O)C(N)C(=O)O', 'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',
        'Y': 'NC(Cc1ccc(O)cc1)C(=O)O', 'V': 'CC(C)C(N)C(=O)O'
    }
    smiles_list = [aa_to_smiles_dict.get(aa.upper(), '') for aa in sequence]
    return '.'.join([s for s in smiles_list if s])

def get_atom_features(atom):
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        atom.GetIsAromatic(),
        int(atom.GetHybridization()),
        atom.GetNumImplicitHs(),
        int(atom.GetChiralTag()),
        len(atom.GetNeighbors()),
        atom.IsInRing(),
        atom.GetMass(),
        atom.GetTotalValence()
    ]
    return features


def mol_to_graph(mol):
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edge indices and features
    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

        feature = [
            bond.GetBondTypeAsDouble(),
            bond.GetIsConjugated(),
            bond.GetIsAromatic()
        ]
        edge_features.extend([feature, feature])

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Combined Dataset class
class CombinedDataset(Dataset):
    def __init__(self, sequences, labels, seq_length=50):
        self.sequences = sequences
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Process sequence data
        sequence_int = aa_to_int(sequence)
        if len(sequence_int) > self.seq_length:
            sequence_int = sequence_int[:self.seq_length]
        else:
            sequence_int = sequence_int + [0] * (self.seq_length - len(sequence_int))
        sequence_tensor = torch.tensor(sequence_int, dtype=torch.long)

        # Process graph data
        smiles = aa_to_smiles(sequence)
        mol = Chem.MolFromSmiles(smiles)
        graph_data = mol_to_graph(mol)

        # Convert label
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return {
            'sequence': sequence_tensor,
            'graph': graph_data,
            'label': label_tensor
        }


def get_data_loader(data_path, batch_size, seq_length=50, shuffle=True):
    dataset = pd.read_csv(data_path)
    combined_data = CombinedDataset(
        sequences=dataset.SEQUENCE.astype(str).values,
        labels=dataset.label.values,
        seq_length=seq_length
    )

    # collate function to handle both graph and sequence data
    def collate_fn(batch):
        sequences = torch.stack([item['sequence'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        graphs = [item['graph'] for item in batch]

        # Create batched graph data
        batch_idx = []
        x_list = []
        edge_index_list = []
        edge_attr_list = []

        for i, graph in enumerate(graphs):
            num_nodes = graph.x.size(0)
            batch_idx.extend([i] * num_nodes)
            x_list.append(graph.x)

            if graph.edge_index.size(1) > 0:
                # Adjust edge indices for batching
                edge_index = graph.edge_index.clone()
                edge_index[0] += sum(g.x.size(0) for g in graphs[:i])
                edge_index[1] += sum(g.x.size(0) for g in graphs[:i])
                edge_index_list.append(edge_index)
                edge_attr_list.append(graph.edge_attr)
        # Combine graph data
        if len(edge_index_list) > 0:
            batched_x = torch.cat(x_list, dim=0)
            batched_edge_index = torch.cat(edge_index_list, dim=1)
            batched_edge_attr = torch.cat(edge_attr_list, dim=0)
        else:
            batched_x = torch.cat(x_list, dim=0)
            batched_edge_index = torch.zeros((2, 0), dtype=torch.long, device=sequences.device)
            batched_edge_attr = torch.zeros((0, 3), dtype=torch.float, device=sequences.device)

        batch_idx_tensor = torch.tensor(batch_idx, dtype=torch.long)

        return {
            'sequences': sequences,
            'x': batched_x,
            'edge_index': batched_edge_index,
            'edge_attr': batched_edge_attr,
            'batch': batch_idx_tensor,
            'labels': labels
        }

    return DataLoader(combined_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)