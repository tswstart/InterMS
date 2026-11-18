from pathlib import Path
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.nn import GATv2Conv, Linear, SAGEConv, to_hetero

from src.utils import penalize_false_negative
from torch.nn import Embedding
from src.layers import MLP, ProtGNNSelector, LigandGNNSelector, ResidueCliqueDiffPoolCoAttention, ResidueCliqueDiffPoolCoAttentionRes, MultiHeadAttentionPooling, DynamicWeightedConcatFusion
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter

class SAGEEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, final_activation, with_linears, in_channels=-1):
        super().__init__()
        self.final_activation = final_activation

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv((-1, -1), hidden_channels, project=with_linears))

        self.lastconv = SAGEConv((-1, -1), out_channels, project=with_linears)

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
        x = self.lastconv(x, edge_index)

        if self.final_activation:
            x = x.relu()

        return x


class GATEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, final_activation, with_linears):
        super().__init__()
        self.with_linears = with_linears
        self.final_activation = final_activation

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv((-1, -1), hidden_channels, edge_dim=1, add_self_loops=False))

        self.lastconv = GATv2Conv((-1, -1), out_channels, edge_dim=1, add_self_loops=False)

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = x.relu()
        x = self.lastconv(x, edge_index, edge_attr)

        if self.final_activation:
            x = x.relu()
        return x


class RegreEdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, link_type, use_dropout=False, use_batchnorm=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        if use_batchnorm:
            self.batchnorm1 = BatchNorm1d(hidden_channels)
        if use_dropout:
            self.dropout1 = Dropout(0.2)
        self.lin2 = Linear(hidden_channels, 1)

        self.link_from = link_type[0]
        self.link_to = link_type[1]

    def forward(self, z):
        z = self.lin1(z)
        if self.use_batchnorm:
            z = self.batchnorm1(z)
        z = z.relu()
        if self.use_dropout:
            z = self.dropout1(z)
        z = self.lin2(z)
        return F.softplus(z)  # smooth and stable

class BinaryEdgeDecoder(nn.Module):
    def __init__(self, hidden_channels, link_type, use_dropout=False, use_batchnorm=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        # Input is the concatenation of two entity vectors, dimension is 2 * hidden_channels
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        if use_batchnorm:
            self.batchnorm1 = BatchNorm1d(hidden_channels)
        if use_dropout:
            self.dropout1 = Dropout(p=0.2)

        self.lin2 = Linear(hidden_channels, 1)  # outputs logits, no activation
        self.link_from = link_type[0]
        self.link_to = link_type[1]

    def forward(self, z):
        z = self.lin1(z)
        if self.use_batchnorm:
            z = self.batchnorm1(z)
        z = torch.relu(z)
        if self.use_dropout:
            z = self.dropout1(z)

        logits = self.lin2(z)  # output shape = (num_edges,1)
        return logits  # Note: do not add sigmoid! Use BCEWithLogitsLoss!

class MultiClassEdgeDecoder(nn.Module):
    def __init__(self, hidden_channels, link_type, num_classes=3, use_dropout=False, use_batchnorm=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.num_classes = num_classes

        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        if use_batchnorm:
            self.batchnorm1 = BatchNorm1d(hidden_channels)
        if use_dropout:
            self.dropout1 = Dropout(p=0.2)

        self.lin2 = Linear(hidden_channels, num_classes)  # outputs logits with shape (num_edges, num_classes)
        self.link_from = link_type[0]
        self.link_to = link_type[1]

    def forward(self, z):
        z = self.lin1(z)
        if self.use_batchnorm:
            z = self.batchnorm1(z)
        z = torch.relu(z)
        if self.use_dropout:
            z = self.dropout1(z)

        logits = self.lin2(z)  # shape: [num_edges, num_classes]
        return logits  # Note: do not add softmax!  

class Model(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        metadata,
        link_type,
        protein_gm,
        compound_gm,
        loss_function="cross_entropy",
        gnn_type="gcn",
        local_num_heads=4,
        local_diffpool_clusters=30,
        global_encoder_type="sage",
        decoder="regression",
        num_encoding_layers=2,
        aggregation_operator="mean",
        final_activation=True,
        with_linears=False,
        use_dropout=False,
    ):
        super().__init__()

        self.protein_gm = protein_gm
        self.compound_gm = compound_gm
        # MOLECULE IN FEAT
        self.atom_type_encoder = Embedding(20,hidden_channels)
        self.atom_feat_encoder = MLP([43, hidden_channels * 2, hidden_channels], out_norm=True) 

        # Clique IN feat
        self.clique_encoder = Embedding(4, hidden_channels)
        self.lin_proj = torch.nn.Linear(hidden_channels, hidden_channels)

        # PROTEIN IN FEAT
        self.prot_evo = MLP([1280, hidden_channels * 2, hidden_channels], out_norm=True) 
        self.prot_aa = MLP([33, hidden_channels * 2, hidden_channels], out_norm=True) 

        self.mol_convs = ProtGNNSelector(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=None,     # your edge feature dimension
            gnn_type=gnn_type,# prefer GAT for types that support edge_attr
            num_layers=2,
            dropout=0.2
        )

        self.pro_convs = LigandGNNSelector(
            in_channels=hidden_channels,     # node feature dimension, e.g. 74-dim RDKit atom features
            out_channels=hidden_channels,   # output embedding
            edge_dim=None,         # edge feature dimension, e.g. bond type, distance, cis/trans isomerism
            gnn_type=gnn_type,     # GIN or GAT recommended for small molecules
            num_layers=2,
            dropout=0.2
        )

        # local encoder
        if decoder == "regression":
            self.local_encoder = ResidueCliqueDiffPoolCoAttention(
                hidden_dim=hidden_channels,
                num_heads=local_num_heads,
                diffpool_clusters=local_diffpool_clusters
            )
        else:
            self.local_encoder = ResidueCliqueDiffPoolCoAttentionRes(
                hidden_dim=hidden_channels,
                num_heads=local_num_heads,
                diffpool_clusters=local_diffpool_clusters,
            )

        # local protein and ligand pooling
        self.protein_pooling = MultiHeadAttentionPooling(hidden_dim=hidden_channels, num_heads=local_num_heads, attn_dropout=0.2)
        self.ligand_pooling = MultiHeadAttentionPooling(hidden_dim=hidden_channels, num_heads=local_num_heads, attn_dropout=0.2)
        self.weighted_concat_protein = DynamicWeightedConcatFusion(dim=hidden_channels)
        self.weighted_concat_ligand = DynamicWeightedConcatFusion(dim=hidden_channels)

        # global encoder
        if global_encoder_type == "sage":
            self.global_encoder = SAGEEncoder(
                hidden_channels,
                hidden_channels,
                num_layers=num_encoding_layers,
                final_activation=final_activation,
                with_linears=with_linears,
            )
            self.global_encoder = to_hetero(self.global_encoder, metadata, aggr=aggregation_operator)
        elif global_encoder_type == "gat":
            self.global_encoder = GATEncoder(
                hidden_channels,
                hidden_channels,
                num_layers=num_encoding_layers,
                final_activation=final_activation,
                with_linears=with_linears,
            )
            self.global_encoder = to_hetero(self.global_encoder, metadata, aggr=aggregation_operator)
        else:
            raise ValueError("Encoder type {} not implemented!".format(global_encoder_type))
        
        if decoder == "regression":
            self.decoder = RegreEdgeDecoder(
                hidden_channels, link_type=link_type, use_dropout=use_dropout
            )
        elif decoder == "binary":
            self.decoder = BinaryEdgeDecoder(
                hidden_channels, link_type=link_type, use_dropout=use_dropout
            )
        elif decoder == "multiclass":
            self.decoder = MultiClassEdgeDecoder(
                hidden_channels, link_type=link_type, use_dropout=use_dropout
            )
        else:
            raise ValueError("Decoder type {} not implemented!".format(decoder))
            
        self.link_from = link_type[0]
        self.link_to = link_type[1]

        # init loss
        if loss_function == "mse":
            self.loss_function = nn.MSELoss()
        elif loss_function == "bce":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif loss_function == "ce":
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError("Loss function {} not implemented!".format(loss_function))

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_attr_dict):
        z_dict = self.global_encoder(x_dict, edge_index_dict, edge_attr_dict)
        return self.decoder(z_dict, edge_label_index)

    def inspect(self, data, mode="all"):
        edge_type = (self.link_from, self.link_to)
        if mode == "train":
            return data[edge_type].y_edge_index[:, data[edge_type].train_idx]
        elif mode == "val":
            return data[edge_type].y_edge_index[:, data[edge_type].val_idx]
        elif mode == "test":
            return data[edge_type].y_edge_index[:, data[edge_type].test_idx]
        elif mode == "all":
            return data[edge_type].y_edge_index
        else:
            raise ValueError("GNN inspect(): mode can only be 'train', 'val', 'test' or 'all'")

    def ground_truth(self, data, mode="all"):
        edge_type = (self.link_from, self.link_to)
        if mode == "train":
            return data[edge_type].y[data[edge_type].train_idx]
        elif mode == "val":
            return data[edge_type].y[data[edge_type].val_idx]
        elif mode == "test":
            return data[edge_type].y[data[edge_type].test_idx]
        elif mode == "all":
            return data[edge_type].y
        else:
            raise ValueError("GNN ground_truth(): mode can only be 'train', 'val', 'test' or 'all'")


    def get_local_pairwise_indices(self, data):

        pro_ptr = data.pro_batch.ptr           # shape=[G+1]
        lig_ptr = data.lig_batch.clique_x_ptr  # you need to add .clique_x_ptr when constructing the Batch
        # get label edge index
        src = data[(self.link_from, self.link_to)].edge_label_index[0]  # Protein endpoint [E]
        dst = data[(self.link_from, self.link_to)].edge_label_index[1]  # Ligand endpoint  [E]

        res_offsets  = pro_ptr[src]
        cli_offsets  = lig_ptr[dst]
        res_lens     = pro_ptr[src+1] - pro_ptr[src]
        cli_lens     = lig_ptr[dst+1] - lig_ptr[dst]

        # ---------- residue ----------
        max_r = int(res_lens.max())                      # max residue number
        range_r = torch.arange(max_r, device=src.device)   # [0,1,…]

        # mask:  True for really exist residue indices
        mask_r  = range_r.unsqueeze(0) < res_lens.unsqueeze(1)   # shape = [E, max_r]

        local_residue_indices = (res_offsets.unsqueeze(1) + range_r) \
                                [mask_r]                          # flatten to 1-D

        # ---------- clique ----------
        max_c = int(cli_lens.max())
        range_c = torch.arange(max_c, device=src.device)
        mask_c  = range_c.unsqueeze(0) < cli_lens.unsqueeze(1)

        local_clique_indices = (cli_offsets.unsqueeze(1) + range_c)[mask_c]
                

        local_residue_batch = torch.repeat_interleave(
            torch.arange(src.size(0), device=dst.device), 
            res_lens                               
        )
        local_clique_batch = torch.repeat_interleave(
            torch.arange(dst.size(0), device=dst.device),
            cli_lens
        )

        return local_residue_indices, local_clique_indices, local_residue_batch, local_clique_batch

    
    def get_small_graph_dict(self, data):

        # ====== Step 1: get small graph ======
        batch_small_graph_pro = data.pro_batch
        batch_small_graph_com = data.lig_batch
       
        # PROTEIN Featurize
        residue_x = self.prot_aa(batch_small_graph_pro.prot_node_aa) + self.prot_evo(batch_small_graph_pro.prot_node_evo)
        # MOLECULE Featurize
        atom_x = self.atom_type_encoder(batch_small_graph_com.atom_idx.squeeze()) + self.atom_feat_encoder(batch_small_graph_com.atom_x_feat)
        # Clique Featurize
        clique_x = self.clique_encoder(batch_small_graph_com.clique_x.squeeze())

        # protein and ligand at residue-level and atom-level feature extraction
        residue_x = self.pro_convs(residue_x, batch_small_graph_pro.prot_edge_index)
        atom_x = self.mol_convs(atom_x, batch_small_graph_com.atom_edge_index)
        # get global feature
        residue_x_pool = global_mean_pool(residue_x, batch_small_graph_pro.batch)
        atom_x_pool = global_mean_pool(atom_x, batch_small_graph_com.batch)
                                       
        # residual connection + atom2clique
        row, col = batch_small_graph_com.atom2clique_index
        hx_clique = scatter(atom_x[row], col, dim=0, dim_size=clique_x.size(0), reduce='mean')
        clique_x = clique_x + F.relu(self.lin_proj(hx_clique))

        x_dict_pooling={}
        x_dict_pooling['target'] = residue_x_pool
        x_dict_pooling['compound'] = atom_x_pool

        return residue_x, atom_x, clique_x, x_dict_pooling


    def encode_graph(self, data, mode):

        # ====== Step 1: get small graph ======
        residue_x, atom_x, clique_x, x_dict_pooling = self.get_small_graph_dict(data)

        # ====== Step 2: get label edge index ======
        src_nodes = data[(self.link_from, self.link_to)].edge_label_index[0]
        dst_nodes = data[(self.link_from, self.link_to)].edge_label_index[1]
        num_edges = src_nodes.size(0)

        # ====== Step 3: get global interaction embedding ======
        global_z_dict = self.global_encoder(x_dict_pooling, data.edge_index_dict, data.edge_attr_dict)
        global_z_src = global_z_dict[self.link_from][src_nodes]
        global_z_dst = global_z_dict[self.link_to][dst_nodes]

        # ====== Step 4: prepare batched local indices ======
        local_residue_indices, local_clique_indices, local_residue_batch_idx, local_clique_batch_idx = self.get_local_pairwise_indices(data)
        
        # ====== Step 5: select + local encoder ======
        local_residue_x = residue_x.index_select(0, local_residue_indices)
        local_clique_x = clique_x.index_select(0, local_clique_indices)

        local_residue_x_pooling, local_residue_batch_idx_pooling, local_clique_x = self.local_encoder(
            local_residue_x, local_residue_batch_idx, local_clique_x, local_clique_batch_idx
        )

        # ====== Step 6: attention pooling ======
        local_z_src, local_protein_attn_raw = self.protein_pooling(local_residue_x_pooling, local_residue_batch_idx_pooling)
        local_z_dst, local_ligand_attn_raw = self.ligand_pooling(local_clique_x, local_clique_batch_idx)

        # ====== Step 7: weighted fusion ======
        final_protein_z, fusion_weights_protein = self.weighted_concat_protein(global_z_src, local_z_src)
        final_ligand_z, fusion_weights_ligand = self.weighted_concat_ligand(global_z_dst, local_z_dst)
        
        # ====== Step 8: concatenate final embeddings ======
        z = torch.cat([final_protein_z, final_ligand_z], dim=-1)

        if mode == "train":
            return z
        elif mode == "test":
            return z, fusion_weights_protein, fusion_weights_ligand
        else:   
            raise ValueError("encode_graph(): mode can only be 'train' or 'test'")

    def train_graph(self, data, penalize_fn=False):

        # 1 encode graph
        z = self.encode_graph(data, mode="train")

        # 2 decode graph
        pred = self.decoder(z)
        target = data[(self.link_from, self.link_to)].edge_label

        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            target = target.squeeze(-1).long()
        elif isinstance(self.loss_function, nn.BCEWithLogitsLoss):
            target = target.float()

        # 3 calculate loss
        loss = self.loss_function(pred, target)
        if penalize_fn:
            loss = penalize_false_negative(loss, pred, target)

        return loss

    @torch.no_grad()
    def test_predict_graph(self, data):
        self.eval()
       
        # 1 encode graph
        z, fusion_weights_protein, fusion_weights_ligand = self.encode_graph(data, mode="test")
        # 2 decode graph
        pred = self.decoder(z)
        target = data[(self.link_from, self.link_to)].edge_label

        return pred, target, z, fusion_weights_protein, fusion_weights_ligand
    
    @torch.no_grad()
    def explain_local(self, data, save_dir: str = None,
                      explain_protein: bool = True,
                      explain_ligand: bool = True,
                      normalize: str = "minmax",
                      norm_clusters: bool = True,
                      eps: float = 1e-8,
                      ligand_reduce: str = "atom"):
        """
        Unified entry point: perform local explanation for protein and ligand as needed.
        """
        results = {}
        if explain_protein:
            protein_save_dir = os.path.join(save_dir, "protein") if save_dir is not None else None
            results["protein"] = self.explain_protein_local(
                data,
                save_dir=protein_save_dir,
                normalize=normalize,
                norm_clusters=norm_clusters,
                eps=eps,
            )
        if explain_ligand:
            ligand_save_dir = os.path.join(save_dir, "ligand") if save_dir is not None else None
            results["ligand"] = self.explain_ligand_local(
                data,
                save_dir=ligand_save_dir,
                normalize=normalize,
                norm_clusters=norm_clusters,
                eps=eps,
                reduce=ligand_reduce,
            )
        return results


    @torch.no_grad()
    def explain_protein_local(self, data, save_dir: str = None,
                    normalize: str = "minmax",
                    norm_clusters: bool = True,
                    eps: float = 1e-8):
        """
        normalize:  residue-level normalization method:
            - "minmax"  : (x - min)/(max - min + eps)  → [0,1]  [default, good for heatmaps/Top-k]
            - "softmax" : softmax(x) → (0,1), sums to 1   [good as a probability distribution]
            - "sigmoid" : sigmoid( zscore(x) ) → (0,1) [robust, insensitive to outliers]
            - "none"    : no normalization (returns raw)
        norm_clusters: whether to apply softmax to cluster-level scores (per sample); True is usually more intuitive
        save_dir:     if given, automatically writes a per-residue explanation CSV for each edge, with scores written to importance_01
        """
        self.eval()
        device = next(self.parameters()).device

        # ====== Small graphs & features ======
        residue_x, atom_x, clique_x, x_dict_pooling = self.get_small_graph_dict(data)

        # ====== Edge index (E P-L pairs in this batch) ======
        edge_type = (self.link_from, self.link_to)
        edge_index = data[edge_type].edge_label_index
        src_nodes = edge_index[0]  # [E] protein idx in this batch
        dst_nodes = edge_index[1]  # [E] ligand  idx in this batch
        E = src_nodes.size(0)

        # ====== Prepare pair-wise residue/substructure indices and batch mapping ======
        (local_residue_indices, local_clique_indices,
        local_residue_batch_idx, local_clique_batch_idx) = self.get_local_pairwise_indices(data)

        local_residue_x = residue_x.index_select(0, local_residue_indices)
        local_clique_x  = clique_x.index_select(0,  local_clique_indices)

        # ====== Local encoder (with S and r_mask) ======
        out = self.local_encoder(local_residue_x, local_residue_batch_idx,
                                local_clique_x,  local_clique_batch_idx,
                                return_assign=True)
        (local_residue_x_pooling,
        local_residue_batch_idx_pooling,
        _local_clique_x,
        S, r_mask) = out  # S: [E, N_r*, K], r_mask: [E, N_r*]

        # ====== Multi-head attention pooling to get attention for each "cluster" ======
        # returns attn_raw: [E*K, num_heads] - multi-head attention for each local cluster
        local_z_src, local_protein_attn_raw = self.protein_pooling(
            local_residue_x_pooling, local_residue_batch_idx_pooling
        )
        # average over heads to get a single importance value per cluster
        cluster_w = local_protein_attn_raw.mean(dim=1)  # [E*K]
        K = self.local_encoder.diffpool_clusters

        # reshape [E*K] back to [E, K]
        cluster_id = torch.arange(K, device=cluster_w.device).repeat(E)  # [E*K]
        flat_index = local_residue_batch_idx_pooling * K + cluster_id    # [E*K]
        cluster_scores = torch.zeros(E * K, device=cluster_w.device)
        cluster_scores.index_add_(0, flat_index, cluster_w)
        cluster_scores = cluster_scores.view(E, K)  # [E, K]

        # (optional) apply softmax to cluster-level scores so they form a distribution over the K clusters of each sample (0-1, sum=1)
        if norm_clusters:
            cluster_scores = torch.softmax(cluster_scores, dim=1)  # [E, K]

        # ====== Propagate back to residues via DiffPool's S: raw_res_scores = S @ cluster_scores[e] ======
        pro_ptr = data.pro_batch.ptr
        res_offsets = pro_ptr[src_nodes]
        res_lens    = pro_ptr[src_nodes + 1] - pro_ptr[src_nodes]

        # batch metadata
        prot_seqs = list(getattr(data.pro_batch, "prot_seq", []))
        lig_smiles_list = list(getattr(data.lig_batch, "lig_smiles", []))

        # predictions and ground truths in this batch (aligned with edge order)
        pred, target, *_ = self.test_predict_graph(data)    # [E] or [E,1]
        pred_flat   = pred.detach().view(-1).cpu().tolist()
        target_flat = target.detach().view(-1).cpu().tolist()

        # containers
        res_positions, res_scores_raw, res_scores_01 = [], [], []
        edges = []
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        src_cpu = src_nodes.detach().cpu()
        dst_cpu = dst_nodes.detach().cpu()

        # -- define a unified 0-1 normalization function
        def normalize_vec(x: torch.Tensor, how: str):
            if how == "none":
                return x
            if how == "minmax":
                xmin, xmax = x.min(), x.max()
                return (x - xmin) / (xmax - xmin + eps)
            if how == "softmax":
                return torch.softmax(x, dim=0)
            if how == "sigmoid":
                # z-score then sigmoid, more numerically robust
                mu, std = x.mean(), x.std()
                z = (x - mu) / (std + eps)
                return torch.sigmoid(z)
            # default fallback
            return x

        for e in range(E):
            valid = r_mask[e]          # [N_r*]
            S_e   = S[e][valid]        # [Nr_e, K]
            raw_e = S_e @ cluster_scores[e]  # [Nr_e] residue-level raw score (unnormalized)

            off = int(res_offsets[e].item())
            ln  = int(res_lens[e].item())
            pos = data.pro_batch.prot_node_pos[off: off + ln].view(-1)  # [Nr_e] (assumed 0-based)

            # normalize to 0-1
            norm_e = normalize_vec(raw_e, normalize)

            res_positions.append(pos.detach().cpu())
            res_scores_raw.append(raw_e.detach().cpu())
            res_scores_01.append(norm_e.detach().cpu())

            # assemble the aggregated result for each edge (with 0-1 scores)
            pro_idx = int(src_cpu[e].item())
            lig_idx = int(dst_cpu[e].item())
            seq = prot_seqs[pro_idx] if pro_idx < len(prot_seqs) else ""
            smiles = lig_smiles_list[lig_idx] if lig_idx < len(lig_smiles_list) else ""

            res_idx_np = pos.detach().cpu().numpy().astype(int)
            aa_chars = [seq[j] if 0 <= j < len(seq) else "" for j in res_idx_np]

            edge_dict = {
                "protein_idx": pro_idx,
                "ligand_idx":  lig_idx,
                "prot_seq": seq,
                "lig_smiles": smiles,
                "pred": float(pred_flat[e]),
                "target": float(target_flat[e]),
                "residue_idx":  res_idx_np,
                "residue_aa":   aa_chars,
                "importance_raw": res_scores_raw[-1].numpy(),  # raw values
                "importance_01":  res_scores_01[-1].numpy(),   # after 0-1 scaling
            }
            edges.append(edge_dict)

            if save_dir is not None:
                import numpy as np
                df = pd.DataFrame({
                    "protein_idx_in_batch": pro_idx,
                    "ligand_idx_in_batch":  lig_idx,
                    "residue_idx":  res_idx_np,
                    "residue_aa":   aa_chars,
                    "importance_01": res_scores_01[-1].numpy(),
                    "importance_raw": res_scores_raw[-1].numpy(),
                    "pred": float(pred_flat[e]),
                    "target": float(target_flat[e]),
                    "lig_smiles": smiles
                })
                df.to_csv(os.path.join(save_dir, f"sample_{e:03d}_residue_importance.csv"),
                        index=False)

        # -- backward compatibility + new fields
        return {
            "protein_idx": src_cpu,
            "ligand_idx":  dst_cpu,
            "res_positions": res_positions,     # list[Tensor]
            "res_scores":    res_scores_raw,    # list[Tensor]  (keeps the original key name, raw)
            "res_scores_01": res_scores_01,     # list[Tensor]  (new, 0-1)
            "edges":         edges              # aggregated view (includes importance_raw/01)
        }
    
    
    @torch.no_grad()
    def explain_ligand_local(self, data,
                            save_dir: str = None,
                            normalize: str = "minmax",
                            norm_clusters: bool = True,  # kept for API compatibility, unused in this implementation
                            eps: float = 1e-8,
                            reduce: str = "atom"  # "atom" or "clique"
                            ):
        """
        Output: ligand-side importance (atom-level by default).
        Requires lig_batch to provide:
        - lig_smiles: list[str] (length = number of ligands)
        - atom2clique_index: LongTensor[2,M], row 0 is the global atom index, row 1 is the global clique index
        - atom_num_nodes: LongTensor[num_lig], number of atoms per ligand
        - clique_num_nodes: LongTensor[num_lig], number of cliques per ligand
        Does not require: ligand-side S_c/c_mask; variable-length cliques are handled slice by slice per sample.
        """
        self.eval()
        device = next(self.parameters()).device

        # ====== Small graph features ======
        residue_x, atom_x, clique_x, x_dict_pooling = self.get_small_graph_dict(data)

        # ====== Edge index (E edges) ======
        edge_type = (self.link_from, self.link_to)
        edge_index = data[edge_type].edge_label_index
        src_nodes = edge_index[0]  # [E] protein idx in this batch
        dst_nodes = edge_index[1]  # [E] ligand  idx in this batch
        E = src_nodes.size(0)

        # ====== pair-wise local index mapping ======
        (local_residue_indices, local_clique_indices,
        local_residue_batch_idx, local_clique_batch_idx) = self.get_local_pairwise_indices(data)

        local_residue_x = residue_x.index_select(0, local_residue_indices)
        local_clique_x  = clique_x.index_select(0,  local_clique_indices)

        # ====== local encoder: only (residue-side) S/r_mask needed; ligand side needs no S_c/c_mask ======
        out = self.local_encoder(local_residue_x, local_residue_batch_idx,
                                local_clique_x,  local_clique_batch_idx,
                                return_assign=True)
        if len(out) >= 5:
            (local_residue_x_pooling,
            local_residue_batch_idx_pooling,
            local_clique_x_pooling,
            S_r, r_mask, *_) = out
            # batch indices of pooled ligand entries are consistent with the pair-wise construction
            local_clique_batch_idx_pooling = local_clique_batch_idx
        else:
            raise RuntimeError("local_encoder(return_assign=True) should return at least 5 items (including residue-side S/r_mask)")

        # ====== ligand pooling: attention for each pooled clique (variable length) ======
        # interface consistent with protein_pooling: returns (pooled, attn_raw=[N_pooled_clique, num_heads])
        clique_z, ligand_attn_raw = self.ligand_pooling(local_clique_x_pooling,
                                                        local_clique_batch_idx_pooling)
        clique_w = ligand_attn_raw.mean(dim=1)                       # [N_pooled_clique]
        clique_batch_idx = local_clique_batch_idx_pooling            # [N_pooled_clique] ∈ [0..E-1]

        # ====== helper: normalize to [0,1] ======
        def normalize_vec(x: torch.Tensor, how: str):
            if x.numel() == 0: return x
            if how == "none":    return x
            if how == "minmax":  return (x - x.min()) / (x.max() - x.min() + eps)
            if how == "softmax": return torch.softmax(x, dim=0)
            if how == "sigmoid":
                z = (x - x.mean()) / (x.std() + eps)
                return torch.sigmoid(z)
            return x

        # ====== batch metadata ======
        lig_smiles_list   = list(getattr(data.lig_batch, "lig_smiles", []))
        atom2clique_index = getattr(data.lig_batch, "atom2clique_index", None)
        atom_num_nodes    = getattr(data.lig_batch, "atom_num_nodes", None)
        clique_num_nodes  = getattr(data.lig_batch, "clique_num_nodes", None)

        if atom2clique_index is None or atom_num_nodes is None or clique_num_nodes is None:
            raise ValueError("lig_batch must contain atom2clique_index / atom_num_nodes / clique_num_nodes")

        atom2clique_index = atom2clique_index.to(device)
        atom_counts   = torch.as_tensor(atom_num_nodes,   device=device, dtype=torch.long).view(-1)
        clique_counts = torch.as_tensor(clique_num_nodes, device=device, dtype=torch.long).view(-1)

        # prefix sums → global index offsets (consistent with the concatenation order of lig_batch)
        atom_ptr   = torch.zeros((atom_counts.numel() + 1,),   device=device, dtype=torch.long)
        clique_ptr = torch.zeros((clique_counts.numel() + 1,), device=device, dtype=torch.long)
        atom_ptr[1:]   = torch.cumsum(atom_counts,   dim=0)
        clique_ptr[1:] = torch.cumsum(clique_counts, dim=0)

        # prediction scores
        pred, target, *_ = self.test_predict_graph(data)
        pred_flat   = pred.detach().view(-1).cpu().tolist()
        target_flat = target.detach().view(-1).cpu().tolist()

        # containers
        clique_positions, clique_scores_raw, clique_scores_01 = [], [], []
        atom_positions,  atom_scores_raw,  atom_scores_01  = [], [], []
        edges = []

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        src_cpu = src_nodes.detach().cpu()
        dst_cpu = dst_nodes.detach().cpu()

        # -- process per sample (edge e): cliques are variable-length; atoms are aggregated via atom2clique_index + ptr
        for e in range(E):
            lig_gid = int(dst_cpu[e].item())  # positional index of the "ligand graph" for this edge within lig_batch
            smiles  = lig_smiles_list[lig_gid] if lig_gid < len(lig_smiles_list) else ""

            # clique selection of this sample (edge e) within the pooled sequence
            mask_e  = (clique_batch_idx == e)                  # [N_pooled_clique] → entries of this sample
            raw_cli = clique_w[mask_e]                         # [Nc_e]
            Nc_e    = int(raw_cli.numel())

            # map pooled entries back to "global clique indices" (the concatenation space of lig_batch.clique)
            # local_clique_indices holds the "global clique index" of each entry in local_clique_x
            # aligned with batch_idx, so mask_e can also slice local_clique_indices
            clique_idx_full = local_clique_indices[mask_e]     # [Nc_e], LongTensor (global clique indices)

            # use the actual global clique indices as positions (easier to align with the original small graph); if you want within-ligand 0..Nc_e-1, change to arange(Nc_e)
            clique_pos = clique_idx_full

            # normalize
            cli_norm = normalize_vec(raw_cli, normalize)
            clique_positions.append(clique_pos.detach().cpu())
            clique_scores_raw.append(raw_cli.detach().cpu())
            clique_scores_01.append(cli_norm.detach().cpu())

             # -- scatter clique scores to atoms (using atom2clique_index)
            atom_pos_e, atom_raw_e, atom_norm_e = None, None, None
            if reduce == "atom" and Nc_e > 0:
                # global index range of atoms/substructures for the current ligand
                a_off = int(atom_ptr[lig_gid].item());   a_len = int(atom_counts[lig_gid].item())
                c_off = int(clique_ptr[lig_gid].item()); c_len = int(clique_counts[lig_gid].item())

                # set of "global clique indices" available for this sample (Long, device-aligned)
                clique_idx_full = clique_idx_full.to(device=device, dtype=torch.long)

                # if this sample has no cliques at all (edge case), skip entirely
                if clique_idx_full.numel() > 0:
                    # build a mapping "global clique index -> local index within this sample (0..Nc_e-1)" (using a tensor rather than a hash; never index out of bounds)
                    max_c = int(clique_idx_full.max().item())
                    map_size = max_c + 1
                    # defensive: map_size may be large but that is fine as long as indexing stays in bounds; initialize to -1
                    g2l = torch.full((map_size,), -1, device=device, dtype=torch.long)
                    g2l[clique_idx_full] = torch.arange(Nc_e, device=device, dtype=torch.long)

                    # take the global (atom, clique) relations, ensuring device & dtype consistency
                    c_idx_all = atom2clique_index[1].to(device=device, dtype=torch.long)
                    a_idx_all = atom2clique_index[0].to(device=device, dtype=torch.long)

                    # keep only edges within the current ligand's atom range
                    valid_a = (a_idx_all >= a_off) & (a_idx_all < a_off + a_len)

                    # Step 1: build only the "in-bounds" c mask (to avoid out-of-bounds g2l[c_idx_all])
                    # note: c_idx_all may contain negative/extreme values, filter to [0, map_size) first
                    if map_size > 0:
                        in_range_c = (c_idx_all >= 0) & (c_idx_all < map_size)
                    else:
                        in_range_c = torch.zeros_like(c_idx_all, dtype=torch.bool, device=device)

                    # Step 2: first build a buffer of c2local = -1, then assign via g2l indexing only for in_range_c entries
                    c2local = torch.full((c_idx_all.numel(),), -1, device=device, dtype=torch.long)
                    if in_range_c.any():
                        c2local[in_range_c] = g2l[c_idx_all[in_range_c]]

                    # Step 3: require both "within the current ligand's atom range" & "has a valid local clique index"
                    mem_mask = valid_a & in_range_c & (c2local >= 0)

                    if mem_mask.any():
                        # raw_cli is indexed by local positions under mask_e, and c2local yields exactly those local indices
                        c_loc = c2local[mem_mask]                    # [M_e] ∈ [0..Nc_e-1]
                        a_glb = a_idx_all[mem_mask]                  # [M_e] global atom indices
                        a_loc = a_glb - a_off                        # convert to "within-ligand local atom index" 0..a_len-1

                        # scatter each clique score to its atoms (sum aggregation)
                        atom_raw = torch.zeros(a_len, device=device)
                        atom_raw.index_add_(0, a_loc, raw_cli.index_select(0, c_loc))

                        atom_norm = normalize_vec(atom_raw, normalize)
                        atom_pos_e  = torch.arange(a_len, device=device)  # 0..a_len-1 (within-ligand local index)
                        atom_raw_e  = atom_raw
                        atom_norm_e = atom_norm

                        atom_positions.append(atom_pos_e.detach().cpu())
                        atom_scores_raw.append(atom_raw_e.detach().cpu())
                        atom_scores_01.append(atom_norm_e.detach().cpu())
                    # if clique_idx_full is empty, nothing to aggregate; keep None (clique-level output only)

            # -- assemble return values / save to disk
            pro_idx = int(src_cpu[e].item())
            pred_e  = float(pred_flat[e]); true_e = float(target_flat[e])

            lig_edge = {
                "protein_idx":  pro_idx,
                "ligand_idx":   lig_gid,
                "lig_smiles":   smiles,
                "pred":         pred_e,
                "target":       true_e,
                # clique level (use "global clique indices" to align with the original small graph)
                "clique_idx":             clique_pos.detach().cpu().numpy() if Nc_e > 0 else np.array([], dtype=np.int64),
                "clique_importance_raw":  raw_cli.detach().cpu().numpy()     if Nc_e > 0 else np.array([], dtype=np.float32),
                "clique_importance_01":   cli_norm.detach().cpu().numpy()    if Nc_e > 0 else np.array([], dtype=np.float32),
            }
            if atom_pos_e is not None:
                lig_edge.update({
                    # atoms use "within-ligand local indices" (0..a_len-1), more convenient for RDKit annotation; add another field if global indices are needed
                    "atom_idx_local":       atom_pos_e.detach().cpu().numpy(),
                    "atom_importance_raw":  atom_raw_e.detach().cpu().numpy(),
                    "atom_importance_01":   atom_norm_e.detach().cpu().numpy(),
                })
            edges.append(lig_edge)

            if save_dir is not None:
                # clique CSV
                df_c = pd.DataFrame({
                    "protein_idx_in_batch": pro_idx,
                    "ligand_idx_in_batch":  lig_gid,
                    "clique_idx_global":    lig_edge["clique_idx"],
                    "importance_01":        lig_edge["clique_importance_01"],
                    "importance_raw":       lig_edge["clique_importance_raw"],
                    "pred":                 pred_e,
                    "target":               true_e,
                    "lig_smiles":           smiles
                })
                df_c.to_csv(os.path.join(save_dir, f"sample_{e:03d}_ligand_clique_importance.csv"), index=False)

                # atom CSV (if available)
                if "atom_idx_local" in lig_edge:
                    df_a = pd.DataFrame({
                        "protein_idx_in_batch": pro_idx,
                        "ligand_idx_in_batch":  lig_gid,
                        "atom_idx_local":       lig_edge["atom_idx_local"],
                        "importance_01":        lig_edge["atom_importance_01"],
                        "importance_raw":       lig_edge["atom_importance_raw"],
                        "pred":                 pred_e,
                        "target":               true_e,
                        "lig_smiles":           smiles
                    })
                    df_a.to_csv(os.path.join(save_dir, f"sample_{e:03d}_ligand_atom_importance.csv"), index=False)

        # -- return
        return {
            "protein_idx": src_cpu,
            "ligand_idx":  dst_cpu,
            # clique level
            "clique_positions": clique_positions,   # list[Tensor] (global clique indices)
            "clique_scores":    clique_scores_raw,  # list[Tensor]
            "clique_scores_01": clique_scores_01,   # list[Tensor]
            # atom level (if available)
            "atom_positions":   atom_positions if len(atom_positions) > 0 else None,   # within-ligand local atom indices
            "atom_scores":      atom_scores_raw if len(atom_scores_raw) > 0 else None,
            "atom_scores_01":   atom_scores_01 if len(atom_scores_01) > 0 else None,
            "edges":            edges
        }
