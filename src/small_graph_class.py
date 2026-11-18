from typing import List, Union, Dict
import torch
from torch_geometric.data import Data, Batch

class CompoundSmallGraphManager:
    def __init__(self, compound_graph_dict: Dict[int, dict], cache_transform=True):
        """
        compound_graph_dict: dict[node_id] -> {
            'atom_feature': Tensor/list,
            'atom_idx': Tensor/list,
            'bond_feature': Tensor,
            'x_clique': Tensor,
            'tree_edge_index': Tensor,
            'atom2clique_index': Tensor
        }
        """
        self.compounds = compound_graph_dict
        self.cache_transform = cache_transform

        self.prebuilt_graphs = {}

        if self.cache_transform:
            for nid, v in self.compounds.items():
                # atom
                v['atom_idx'] = v['atom_idx'].long().view(-1, 1)
                v['atom_feature'] = v['atom_feature'].float()
                adj = v['bond_feature'].long()
                mol_edge_index =  adj.nonzero(as_tuple=False).t().contiguous()
                v['atom_edge_index'] = mol_edge_index.long()
                v['atom_edge_attr'] = adj[mol_edge_index[0], mol_edge_index[1]].long()
                v['atom_num_nodes'] = v['atom_idx'].shape[0]

                ## Clique
                v['x_clique'] = v['x_clique'].long().view(-1, 1)
                v['clique_num_nodes'] = v['x_clique'].shape[0]
                v['tree_edge_index'] = v['tree_edge_index'].long()
                v['atom2clique_index'] = v['atom2clique_index'].long()

                data = MultiGraphData(
                    # x
                    atom_idx=v['atom_idx'],
                    atom_x_feat=v['atom_feature'],
                    clique_x=v['x_clique'],
                    # # edge_index
                    atom_edge_index=v['atom_edge_index'],
                    clique_edge_index=v['tree_edge_index'],
                    # attr
                    atom_edge_attr=v['atom_edge_attr'], # this is atom edge
                    # other
                    atom2clique_index=v['atom2clique_index'],
                    atom_num_nodes=v['atom_num_nodes'],
                    clique_num_nodes=v['clique_num_nodes'],
                    lig_smiles=v['smiles']  # string keep as it is
                )

                data.num_nodes = v['atom_num_nodes']
                # cache the graph data
                self.prebuilt_graphs[nid] = data

    def get_batch(self, node_ids: Union[List[int], torch.Tensor]) -> Batch:
        data_list = [self.prebuilt_graphs[int(nid)] for nid in node_ids]
        return Batch.from_data_list(data_list, follow_batch=['atom_idx', 'clique_x'])
    
    
class ProteinSmallGraphManager:
    def __init__(self, protein_graph_dict: Dict[int, dict], cache_transform=True):
        """
        protein_graph_dict: dict[node_id] -> {
            'seq_feat': Tensor,
            'token_representation': Tensor,
            'num_nodes': int,
            'num_pos': Tensor,
            'edge_index': Tensor,
            'edge_weight': Tensor
        }
        """
        self.proteins = protein_graph_dict
        self.cache_transform = cache_transform

        # Prebuild graph data
        self.prebuilt_graphs = {}

        if self.cache_transform:
            for nid, v in self.proteins.items():
                # protein
                v['seq_feat'] = v['seq_feat'].float()
                v['token_representation'] = v['token_representation'].float()
                v['num_nodes'] = len(v['seq'])
                v['node_pos'] = torch.arange(len(v['seq'])).reshape(-1,1)
                v['edge_weight'] = v['edge_weight'].float()

                data = MultiGraphData(
                    # x
                    prot_node_aa=v['seq_feat'],
                    prot_node_evo=v['token_representation'],
                    # edge_index
                    prot_edge_index=v['edge_index'],
                    # other
                    prot_edge_weight=v['edge_weight'], # edge weight
                    prot_num_nodes=v['num_nodes'], # number of residues
                    prot_node_pos=v['node_pos'], # residue positions 0,1,2,3...
                    prot_seq=v['seq']  # keep the string as is
                )
                # cache the graph data
                self.prebuilt_graphs[nid] = data

    def get_batch(self, node_ids: Union[List[int], torch.Tensor]) -> Batch:
        data_list = [self.prebuilt_graphs[int(nid)] for nid in node_ids]
        return Batch.from_data_list(data_list, follow_batch=['prot_node_aa'])
 
class MultiGraphData(Data):
    def __inc__(self, key, item, *args):
        if key == 'atom_edge_index':
            return self.atom_idx.size(0)
        elif key == 'clique_edge_index':
            return self.clique_x.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.atom_idx.size(0)], [self.clique_x.size(0)]])
        elif key == 'prot_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'prot_struc_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'm2p_edge_index':
             return torch.tensor([[self.atom_idx.size(0)], [self.prot_node_aa.size(0)]])
        # elif key == 'edge_index_p2m':
        #     return torch.tensor([[self.prot_node_s.size(0)],[self.atom_idx.size(0)]])
        else:
            return super(MultiGraphData, self).__inc__(key, item, *args)


if __name__ == '__main__':

    protein_dict = torch.load('data/1.pdbbind/protein_graphs.pt')
    ligand_dict = torch.load('data/1.pdbbind/ligand_graphs.pt')

    pgm = ProteinSmallGraphManager(protein_dict)
    cgm = CompoundSmallGraphManager(ligand_dict)

    b = pgm.get_batch([0, 1, 7])
    c = cgm.get_batch([0, 1, 9])

    print(b)
    print(c)