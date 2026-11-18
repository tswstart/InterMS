import torch.nn as nn
import torch.nn.functional as F
import torch

from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv, GINConv

from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter


class MLP(nn.Module):

    def __init__(self, dims, out_norm=False, in_norm=False, bias=True): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear(dims[idx-1], dims[idx], bias=bias) for idx in range(1,len(dims)) ]
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for idx in range(self.hidden_layers+1):
            self.FC_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x):
        y = x
        # Input Layer Norm
        if self.in_norm:
            y = self.in_ln(y)

        for idx in range(self.hidden_layers):
            y = self.FC_layers[idx](y)
            y = F.relu(y)
        y = self.FC_layers[-1](y)

        if self.out_norm:
            y = self.out_ln(y)

        return y


class ProtGNNSelector(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=None, gnn_type='gcn', num_layers=2, dropout=0.2):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.res_linears = nn.ModuleList() if gnn_type == 'gcnres' else None

        # Add an edge_encoder if edge_attr is present
        if edge_dim is not None:
            self.edge_encoder = nn.Linear(edge_dim, out_channels)
        else:
            self.edge_encoder = None

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else out_channels

            if gnn_type == 'gcn' or gnn_type == 'gcnres':
                conv = GCNConv(in_dim, out_channels)
            elif gnn_type == 'gat':
                conv = GATConv(in_dim, out_channels // 4, heads=4, concat=True, edge_dim=edge_dim)
            elif gnn_type == 'sage':
                conv = SAGEConv(in_dim, out_channels)
            elif gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels),
                )
                conv = GINConv(mlp)
            else:
                raise ValueError(f"Unsupported gnn_type: {gnn_type}")

            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(out_channels))

            if gnn_type == 'gcnres':
                self.res_linears.append(nn.Linear(in_dim, out_channels) if in_dim != out_channels else nn.Identity())

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr_encoded = self.edge_encoder(edge_attr)
        else:
            edge_attr_encoded = edge_attr  # may be None

        for i in range(self.num_layers):
            x_in = x
            conv = self.convs[i]

            # Some conv types do not support edge_attr
            if self.gnn_type in ['gat']:
                x = conv(x, edge_index, edge_attr=edge_attr_encoded)
            else:
                x = conv(x, edge_index)

            x = self.bns[i](x)
            x = F.relu(x)

            if self.gnn_type == 'gcnres':
                x = x + self.res_linears[i](x_in)

            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.edge_encoder is not None:
            nn.init.xavier_uniform_(self.edge_encoder.weight)
            if self.edge_encoder.bias is not None:
                nn.init.zeros_(self.edge_encoder.bias)
        if self.gnn_type == 'gcnres':
            for res_lin in self.res_linears:
                if isinstance(res_lin, nn.Linear):
                    nn.init.xavier_uniform_(res_lin.weight)
                    if res_lin.bias is not None:
                        nn.init.zeros_(res_lin.bias)

class LigandGNNSelector(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=None, gnn_type='gin', num_layers=2, dropout=0.2):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if edge_dim is not None:
            self.edge_encoder = nn.Linear(edge_dim, out_channels)
        else:
            self.edge_encoder = None

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else out_channels

            if gnn_type == 'gat':
                conv = GATConv(in_dim, out_channels // 4, heads=4, concat=True, edge_dim=edge_dim)
            elif gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
                conv = GINConv(mlp)
            elif gnn_type == 'sage':
                conv = SAGEConv(in_dim, out_channels)
            elif gnn_type == 'gcn':
                conv = GCNConv(in_dim, out_channels)
            else:
                raise ValueError(f"Unsupported gnn_type: {gnn_type}")

            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(out_channels))

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.num_layers):
            if self.gnn_type == 'gat':
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)

            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.edge_encoder is not None:
            nn.init.xavier_uniform_(self.edge_encoder.weight)
            if self.edge_encoder.bias is not None:
                nn.init.zeros_(self.edge_encoder.bias)
                

class ResidueCliqueDiffPoolCoAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, diffpool_clusters=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.diffpool_clusters = diffpool_clusters

        # DiffPool layer
        self.assign_net = nn.Linear(hidden_dim, diffpool_clusters)
        self.embed_net = nn.Linear(hidden_dim, hidden_dim)

        # Co-Attention layer
        self.proj_residue = nn.Linear(hidden_dim, hidden_dim)
        self.proj_clique = nn.Linear(hidden_dim, hidden_dim)
        self.attn_r2c = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_c2r = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, residue_x, residue_batch, clique_x, clique_batch, return_assign: bool=False):
        """
        residue_x:          [N_r, H]
        residue_batch_idx:  [N_r]
        clique_x:           [N_c, H]
        clique_batch_idx:   [N_c]
        """
        B = torch.max(residue_batch).item() + 1   # number of batches
        device = residue_x.device

        # -------- 1. pad dense batch --------
        r_dense, r_mask = to_dense_batch(residue_x, residue_batch)   # [B, N_r*, H]
        c_dense, c_mask = to_dense_batch(clique_x,  clique_batch)    # [B, N_c*, H]

        # -------- 2. DiffPool on residue --------
        S = self.assign_net(r_dense)                         # [B, N_r*, K]
        S.masked_fill_(~r_mask.unsqueeze(-1), -1e9)
        S = torch.softmax(S, dim=1)                          # normalize over residues within each batch
        r_embed = self.embed_net(r_dense)                    # [B, N_r*, H]
        pooled_r = torch.einsum("bnk,bnh->bkh", S, r_embed)  # [B, K, H]

        # -------- 3. Cross Attention --------
        r_proj = self.proj_residue(pooled_r)                 # [B, K, H]
        c_proj = self.proj_clique(c_dense)                   # [B, N_c*, H]

        # Note: in MultiheadAttention, key_padding_mask=True means **masked = excluded from computation**
        r_out,_ = self.attn_r2c(
            query=r_proj,
            key=c_proj,
            value=c_proj,
            key_padding_mask=~c_mask
        )                                                    # [B, K, H]

        c_out,_ = self.attn_c2r(
            query=c_proj,
            key=r_proj,
            value=r_proj,
        )                                                    # [B, N_c*, H]

        # -------- 4. restore back to sparse format --------
        # (1) residue  –– flatten & build batch_idx
        updated_residue_x = r_out.reshape(-1, self.hidden_dim)            # [B*K, H]
        new_res_batch_idx = torch.arange(B, device=device)\
                               .repeat_interleave(self.diffpool_clusters) # [B*K]

        # (2) clique –– keep only valid nodes
        c_out_flat = c_out.reshape(-1, self.hidden_dim)                   # [B*N_c*, H]
        valid_mask = c_mask.view(-1)                                      # [B*N_c*]
        updated_clique_x = c_out_flat[valid_mask]                         # [N_c, H]

        if return_assign:
            return updated_residue_x, new_res_batch_idx, updated_clique_x, S, r_mask
        else:
            return updated_residue_x, new_res_batch_idx, updated_clique_x
    

class ResidueCliqueDiffPoolCoAttentionRes(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, diffpool_clusters=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.diffpool_clusters = diffpool_clusters

        # DiffPool layer
        self.assign_net = nn.Linear(hidden_dim, diffpool_clusters)
        self.embed_net = nn.Linear(hidden_dim, hidden_dim)

        # QKV projection (6 MLPs -> 2 pairs of QKV)
        self.r_q = nn.Linear(hidden_dim, hidden_dim)
        self.r_k = nn.Linear(hidden_dim, hidden_dim)
        self.r_v = nn.Linear(hidden_dim, hidden_dim)
        self.c_q = nn.Linear(hidden_dim, hidden_dim)
        self.c_k = nn.Linear(hidden_dim, hidden_dim)
        self.c_v = nn.Linear(hidden_dim, hidden_dim)

        self.attn_r2c = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_c2r = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.layernorm_r = nn.LayerNorm(hidden_dim)
        self.layernorm_c = nn.LayerNorm(hidden_dim)

        self.mlp_r = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, residue_x, residue_batch, clique_x, clique_batch, return_assign: bool=False):
        B = torch.max(residue_batch).item() + 1
        device = residue_x.device

        # -------- 1. pad into a dense batch --------
        r_dense, r_mask = to_dense_batch(residue_x, residue_batch)   # [B, N_r*, H]
        c_dense, c_mask = to_dense_batch(clique_x,  clique_batch)    # [B, N_c*, H]

        # -------- 2. DiffPool on residue --------
        S = self.assign_net(r_dense)
        S.masked_fill_(~r_mask.unsqueeze(-1), -1e9)
        S = torch.softmax(S, dim=1)
        r_embed = self.embed_net(r_dense)
        pooled_r = torch.einsum("bnk,bnh->bkh", S, r_embed)         # [B, K, H]

        # -------- 3. Cross Attention --------
        q_r = self.r_q(pooled_r)
        k_c = self.c_k(c_dense)
        v_c = self.c_v(c_dense)

        q_c = self.c_q(c_dense)
        k_r = self.r_k(pooled_r)
        v_r = self.r_v(pooled_r)

        r_attn_out, _ = self.attn_r2c(q_r, k_c, v_c, key_padding_mask=~c_mask)
        c_attn_out, _ = self.attn_c2r(q_c, k_r, v_r)

        r_attn_out = r_attn_out + q_r  # residual connection
        c_attn_out = c_attn_out + q_c  # residual connection
        # LayerNorm
        r_norm = self.layernorm_r(r_attn_out)
        c_norm = self.layernorm_c(c_attn_out)

        # ---------4. project ---------------
        r_mlp = self.mlp_r(r_norm)
        c_mlp = self.mlp_c(c_norm)

        # Combine the original Q with the MLP output
        r_out = r_mlp + r_norm  # residual connection
        c_out = c_mlp + c_norm  # residual connection
        
        # -------- 5. restore back to sparse format --------
        # (1) residue  –– flatten & build batch_idx
        updated_residue_x = r_out.reshape(-1, self.hidden_dim)
        new_res_batch_idx = torch.arange(B, device=device).repeat_interleave(self.diffpool_clusters)
        
        # (2) clique –– keep only valid nodes
        c_out_flat = c_out.reshape(-1, self.hidden_dim)
        valid_mask = c_mask.view(-1)
        updated_clique_x = c_out_flat[valid_mask]

        if return_assign:
            return updated_residue_x, new_res_batch_idx, updated_clique_x, S, r_mask
        else:
            return updated_residue_x, new_res_batch_idx, updated_clique_x


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, attn_dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Independent attention scoring for each head
        self.attn_score = nn.Linear(hidden_dim, num_heads)

        # Final linear fusion
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, batch_idx):
        """
        x: [N_node, hidden_dim]
        batch_idx: [N_node]
        """
        N, _ = x.size()

        # 1. Attention weights (per head)
        attn_raw = torch.sigmoid(self.attn_score(x))  # [N_node, num_heads]

        # 2. Split into multi-head features
        x = x.view(N, self.num_heads, self.head_dim)             # [N, H, D]
        attn_weight = attn_raw.unsqueeze(-1)                     # [N, H, 1]
        weighted_x = x * attn_weight                             # [N, H, D]

        # 3. Flatten to [N, H*D] for scatter aggregation
        weighted_x = weighted_x.view(N, -1)                      # [N, H*D]

        # 4. Scatter-aggregate by batch: weighted sum within each batch
        pooled_x = scatter(weighted_x, batch_idx, dim=0, reduce='sum')  # [B, H*D]

        # 5. Linear fusion
        pooled_x = self.out_proj(pooled_x)                       # [B, hidden_dim]
        pooled_x = self.dropout(pooled_x)

        return pooled_x, attn_raw

class DynamicWeightedConcatFusion(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim // 2

        # Small MLP that dynamically computes fusion weights from global and local features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # outputs two weights: global_weight, local_weight
        )

        # Added: a linear layer that projects the concatenated [2*dim] back to [dim]
        self.output_proj = nn.Linear(dim * 2, dim)

    def forward(self, global_z, local_z):
        """
        global_z: [batch_size, dim]
        local_z:  [batch_size, dim]
        """
        batch_size = global_z.size(0)

        # Concatenate global and local
        concat_z = torch.cat([global_z, local_z], dim=-1)  # [batch_size, 2*dim]

        # Compute weights
        raw_weights = self.fusion_mlp(concat_z)  # [batch_size, 2]
        fusion_weights = torch.softmax(raw_weights, dim=-1)  # softmax normalization

        global_weight = fusion_weights[:, 0].unsqueeze(-1)  # [batch_size, 1]
        local_weight = fusion_weights[:, 1].unsqueeze(-1)   # [batch_size, 1]

        # Weight each branch separately
        weighted_global = global_weight * global_z
        weighted_local = local_weight * local_z

        # Final concatenation
        final_z = torch.cat([weighted_global, weighted_local], dim=-1)  # [batch_size, 2*dim]

        # Added: linear projection back to [batch_size, dim]
        final_z = self.output_proj(final_z)  # [batch_size, dim]

        return final_z, fusion_weights
