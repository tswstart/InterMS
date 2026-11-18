from torch_geometric.loader import LinkNeighborLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import torch.optim as optim
from tqdm import tqdm
from small_graph_class import ProteinSmallGraphManager, CompoundSmallGraphManager
from torch.nn import Embedding
from layers import MLP
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class MolConvs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MolConvs, self).__init__()
        # Graph convolution layers for updating atom features
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, atom_x, atom_edge_index):
        # GCNConv takes node features and edge indices as input
        x = self.conv1(atom_x, atom_edge_index)
        x = torch.relu(x)
        x = self.conv2(x, atom_edge_index)
        return x

class ProtConvs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProtConvs, self).__init__()
        # Graph convolution layers for updating residue features
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, residue_x, residue_edge_index):
        # GCNConv takes node features, edge indices and edge weights as input
        x = self.conv1(residue_x, residue_edge_index)
        x = torch.relu(x)
        x = self.conv2(x, residue_edge_index)
        return x

class GraphModel(nn.Module):
    def __init__(self, mol_in_channels, prot_in_channels, out_channels):
        super(GraphModel, self).__init__()
        # mol_convs and prot_convs handle the molecule graph and the protein graph respectively
        self.mol_convs = MolConvs(mol_in_channels, out_channels)
        self.prot_convs = ProtConvs(prot_in_channels, out_channels)
        
        # Fully connected layer for pooled features
        self.fc = nn.Linear(out_channels * 2, 1)

    def forward(self, batch):
        # Molecule part
        mol_x = self.mol_convs(batch.atom_x, batch.atom_edge_index)
        mol_x_pool = global_mean_pool(mol_x, batch.batch)  # pool over each graph

        # Protein part
        prot_x = self.prot_convs(batch.residue_x, batch.residue_edge_index, batch.residue_edge_attr)
        prot_x_pool = global_mean_pool(prot_x, batch.batch)  # pool over each graph

        # Concatenate the pooled molecule and protein representations
        x = torch.cat([mol_x_pool, prot_x_pool], dim=-1)

        # Final output through the fully connected layer
        out = self.fc(x)
        return out


class HeteroRegressor(nn.Module):
    def __init__(self, metadata, device, hidden_channels=200):
        """
        Args:
            metadata: metadata of HeteroData (node_types, edge_types),
                      can be obtained via data.metadata()
            hidden_channels: hidden dimension of the GNN
        """
        super().__init__()


        self.device = device

        # MOLECULE IN FEAT
        self.atom_type_encoder = Embedding(20,hidden_channels)
        self.atom_feat_encoder = MLP([43, hidden_channels * 2, hidden_channels], out_norm=True) 

        # Clique IN feat
        self.clique_encoder = Embedding(4, hidden_channels)

        # PROTEIN IN FEAT
        self.prot_evo = MLP([1280, hidden_channels * 2, hidden_channels], out_norm=True) 
        self.prot_aa = MLP([33, hidden_channels * 2, hidden_channels], out_norm=True) 

        self.mol_convs = MolConvs(200, 128)
        self.pro_convs = ProtConvs(200, 128)



        # 1) Build HeteroConv, using one SAGEConv for each edge_type
        #    you can also switch to GCNConv, GATConv, etc.
        self.convs = HeteroConv(
            {
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            },
            aggr='mean'  # mean aggregation over updates from the different relations
        )

        # 2) After obtaining (src_emb, dst_emb), apply a regression mapping to them
        #    the MLP structure can be adjusted as needed
        self.fc_edge = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)  # output a scalar
        )

    def forward(self, x_dict, edge_index_dict, target_id, compound_id, 
                target_edge_type, edge_label_index, protein_gm, compound_gm):
        """
        Args:
            x_dict: dict(node_type -> node feature matrix)
            edge_index_dict: dict((src_type, rel, dst_type) -> edge_index)
            target_edge_type: e.g. ('compound', 'interaction', 'target')
            edge_label_index: shape=[2, num_edges], edges to predict in this batch
        Returns:
            predictions: regression results of shape=[num_edges, 1]
        """

        batch_small_graph_pro = protein_gm.get_batch(target_id).to(self.device)
        batch_small_graph_com = compound_gm.get_batch(compound_id).to(self.device)

        # PROTEIN Featurize
        residue_x = self.prot_aa(batch_small_graph_pro.prot_node_aa) + self.prot_evo(batch_small_graph_pro.prot_node_evo)
       
        # MOLECULE Featurize
        atom_x = self.atom_type_encoder(batch_small_graph_com.atom_idx.squeeze()) + self.atom_feat_encoder(batch_small_graph_com.atom_x_feat)

        residue_x = self.pro_convs(residue_x, batch_small_graph_pro.prot_edge_index)
        residue_x_pool = global_mean_pool(residue_x, batch_small_graph_pro.batch)

        atom_x = self.mol_convs(atom_x, batch_small_graph_com.atom_edge_index)
        atom_x_pool = global_mean_pool(atom_x, batch_small_graph_com.batch)

        x_dict['target'] = residue_x_pool
        x_dict['compound'] = atom_x_pool

        # 1) Update the representations of all node types in the large graph
        x_dict = self.convs(x_dict, edge_index_dict)

        # 2) Concatenate the source and target node representations of the edges to regress
        src, dst = edge_label_index
        src_type, _, dst_type = target_edge_type
        src_emb = x_dict[src_type][src]
        dst_emb = x_dict[dst_type][dst]
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)  # [num_edges, 2*hidden_dim]

        # 3) Feed into the MLP to regress the scalar
        out = self.fc_edge(edge_emb)  # shape=[num_edges, 1]
        return out

if __name__ == "__main__":
    
    
    device = torch.device('cuda:7')
    # biggraph
    hetero_data = torch.load('data/pdbbind-data-complete_Averaged_FingerprintsECFP-256_LipinskiFea_ExtChemicalFea_CompSimilarities0.8_3-khmer_GeneExprLung_SubcellularLoc_9606.protein.physical.links.v12.0.txt.pt')
    # smallgraph
    protein_dict = torch.load('data/protein_small_graphs.pt')
    ligand_dict = torch.load('data/ligand_small_graphs.pt')

    protein_gm = ProteinSmallGraphManager(protein_dict)
    compound_gm = CompoundSmallGraphManager(ligand_dict)

    # Extract the full positive edge indices and labels
    full_edge_index = hetero_data['compound', 'interaction', 'target'].y_edge_index  # shape=[2, num_pos_edges]
    full_edge_label = hetero_data['compound', 'interaction', 'target'].y            # shape=[num_pos_edges, 1]

    # Extract the train/validation/test split indices
    train_idx = hetero_data['compound', 'interaction', 'target'].train_idx
    # val_idx   = hetero_data['compound', 'interaction', 'target'].val_idx
    test_idx  = hetero_data['compound', 'interaction', 'target'].test_idx

    # Get the positive edge indices and labels for the train/validation/test splits respectively
    train_edge_index = full_edge_index[:, train_idx]
    train_edge_label = full_edge_label[train_idx]

    # val_edge_index = full_edge_index[:, val_idx]
    # val_edge_label = full_edge_label[val_idx]

    test_edge_index = full_edge_index[:, test_idx]
    test_edge_label = full_edge_label[test_idx]

    del hetero_data["compound"].smiles
    del hetero_data["target"].names
    del hetero_data["compound"].names

    train_loader = LinkNeighborLoader(
        data=hetero_data,
        num_neighbors=[10],  # number of sampled neighbors per layer, e.g. two numbers for a 2-layer GNN
        edge_label_index=(('compound', 'interaction', 'target'), train_edge_index),
        edge_label=train_edge_label,
        # negative sampling ratio, e.g. 1.0 means "one negative sample per positive sample"
        neg_sampling_ratio=0.0,  
        batch_size=4096,
        shuffle=True
    )


    metadata = hetero_data.metadata()  # (node_types, edge_types)
    model = HeteroRegressor(metadata, hidden_channels=200, device=device).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for epoch in tqdm(range(1, 51)):
        model.train()
        total_loss = 0
        
        for batch_data in train_loader:  # train_loader = LinkNeighborLoader(...)
            batch_data = batch_data.to(device)
            
            # 1) Forward pass
            pred = model(
                x_dict=batch_data.x_dict,
                edge_index_dict=batch_data.edge_index_dict,
                target_id=batch_data['target'].n_id,
                compound_id=batch_data['compound'].n_id,
                target_edge_type=('compound', 'interaction', 'target'),  # your target edge type
                edge_label_index=batch_data[('compound', 'interaction', 'target')].edge_label_index,
                protein_gm=protein_gm,
                compound_gm=compound_gm
            )
            
            # 2) Compute the loss
            #    note that batch_data.edge_label may be [batch_size],
            #    it needs to be reshaped to [batch_size,1] or dimension-matched with pred.squeeze(-1)
            label = batch_data[('compound', 'interaction', 'target')].edge_label  # shape=[batch_size] or [batch_size,1]
            if label.dim() == 1:
                label = label.unsqueeze(-1)  # ensure it is [batch_size,1]
            
            loss = loss_fn(pred, label.float())
            
            # 3) Backward pass & optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * label.size(0)
        
        # Compute the average loss of the current epoch
        avg_loss = total_loss / len(train_loader.dataset)  
        print(f"Epoch {epoch} | Loss {avg_loss:.4f}")