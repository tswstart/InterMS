import argparse
import copy
import logging
import os
import sys

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

from src.utils import load_config, parse_params, add_small_graphs
from src.model import Model
from src.dataset import BioSNAPDataset, PDBBindDataset, FunctionDataset, BindingDBDataset
from src.metrics import (
    compute_regression_metrics,
    compute_binary_classification_metrics,
    compute_multiclass_classification_metrics,
)
from src.small_graph_class import ProteinSmallGraphManager, CompoundSmallGraphManager
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DATASET_REGISTRY = {
    'pdbbind2016': PDBBindDataset,
    'pdbbind2020': PDBBindDataset,
    'biosnap': BioSNAPDataset,
    'function': FunctionDataset,
    'bindingdb': BindingDBDataset,
}


def load_dataset(params):
    run_name = params.RUNINFO.run_name
    dataset_cls = DATASET_REGISTRY.get(run_name)
    if dataset_cls is None:
        raise ValueError(
            f"Unknown dataset '{run_name}'. Supported: {list(DATASET_REGISTRY.keys())}"
        )
    dataset = dataset_cls(settings=params, dataset_name=params.RUNINFO.dataset_name)
    data = dataset.get()
    return dataset, data


def init_model(params, data, metadata):
    print("Loading protein graphs...")
    protein_dict = torch.load(
        os.path.join(params.RUNINFO.data_dir, f"{params.RUNINFO.dataset_name}_protein_graphs.pt")
    )
    print("Loading ligand graphs...")
    ligand_dict = torch.load(
        os.path.join(params.RUNINFO.data_dir, f"{params.RUNINFO.dataset_name}_ligand_graphs.pt")
    )

    num_protein_nodes = data['target'].num_nodes
    num_ligand_nodes = data['compound'].num_nodes
    assert len(protein_dict) == num_protein_nodes, \
        f"Mismatch: protein_dict has {len(protein_dict)} entries, but data['target'] has {num_protein_nodes} nodes."
    assert len(ligand_dict) == num_ligand_nodes, \
        f"Mismatch: ligand_dict has {len(ligand_dict)} entries, but data['compound'] has {num_ligand_nodes} nodes."

    protein_gm = ProteinSmallGraphManager(protein_dict)
    compound_gm = CompoundSmallGraphManager(ligand_dict)

    model = Model(
        hidden_channels=params.MODELINFO.hidden_channels,
        metadata=metadata,
        link_type=params.MODELINFO.link_prediction_mode,
        protein_gm=protein_gm,
        compound_gm=compound_gm,
        loss_function=params.RUNPARAMS.loss_function,
        gnn_type=params.MODELINFO.gnn_type,
        local_num_heads=params.MODELINFO.local_encoder_num_heads,
        local_diffpool_clusters=params.MODELINFO.local_diffpool_clusters,
        global_encoder_type=params.MODELINFO.global_encoder_type,
        decoder=params.MODELINFO.decoder_type,
        num_encoding_layers=params.MODELINFO.num_encoding_layers,
        aggregation_operator=params.MODELINFO.conv_aggregation_operator,
        final_activation=params.MODELINFO.final_activation,
        with_linears=params.MODELINFO.conv_with_linears,
        use_dropout=params.MODELINFO.use_edge_decoder_dropout,
    )
    return model


def build_data_loaders(params, data, model, batch_size=None):
    """Build train/val/test LinkNeighborLoaders from hetero graph data."""
    if batch_size is None:
        if isinstance(params.RUNPARAMS.batch_size, int) and params.RUNPARAMS.batch_size > 0:
            batch_size = params.RUNPARAMS.batch_size
        else:
            batch_size = data[params.MODELINFO.link_prediction_mode].y.size(0)

    num_neighbors = [int(x) for x in params.RUNPARAMS.num_neighbors]
    num_workers = int(params.RUNPARAMS.num_workers)

    full_edge_index = data[params.MODELINFO.link_prediction_mode].y_edge_index
    full_edge_label = data[params.MODELINFO.link_prediction_mode].y

    train_idx = data[params.MODELINFO.link_prediction_mode].train_idx
    val_idx = data[params.MODELINFO.link_prediction_mode].val_idx
    test_idx = data[params.MODELINFO.link_prediction_mode].test_idx

    train_edge_index = full_edge_index[:, train_idx]
    train_edge_label = full_edge_label[train_idx]
    val_edge_index = full_edge_index[:, val_idx]
    val_edge_label = full_edge_label[val_idx]
    test_edge_index = full_edge_index[:, test_idx]
    test_edge_label = full_edge_label[test_idx]

    hetero_graph = copy.deepcopy(data)
    del hetero_graph["compound"].smiles
    del hetero_graph["target"].names
    del hetero_graph["compound"].names
    del hetero_graph['compound', 'interaction', 'target'].test_idx
    del hetero_graph['compound', 'interaction', 'target'].train_idx
    del hetero_graph['compound', 'interaction', 'target'].val_idx
    del hetero_graph['target', 'rev_interaction', 'compound'].test_idx
    del hetero_graph['target', 'rev_interaction', 'compound'].train_idx
    del hetero_graph['target', 'rev_interaction', 'compound'].val_idx

    transform_fn = add_small_graphs(model.protein_gm, model.compound_gm)

    train_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=num_neighbors,
        edge_label_index=(params.MODELINFO.link_prediction_mode, train_edge_index),
        edge_label=train_edge_label,
        neg_sampling_ratio=0.0,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        transform=transform_fn,
    )
    val_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=num_neighbors,
        edge_label_index=(params.MODELINFO.link_prediction_mode, val_edge_index),
        edge_label=val_edge_label,
        neg_sampling_ratio=0.0,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        transform=transform_fn,
    )
    test_loader = LinkNeighborLoader(
        data=hetero_graph,
        num_neighbors=num_neighbors,
        edge_label_index=(params.MODELINFO.link_prediction_mode, test_edge_index),
        edge_label=test_edge_label,
        neg_sampling_ratio=0.0,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        transform=transform_fn,
    )
    return train_loader, val_loader, test_loader


def run_pipeline(params, data, model, output_f):
    device = torch.device("cuda" if torch.cuda.is_available() and not params.RUNINFO.force_cpu else "cpu")
    model = model.to(device)

    num_params = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_params:d}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.RUNPARAMS.learning_rate, weight_decay=1e-4)

    train_loader, val_loader, test_loader = build_data_loaders(params, data, model)
    save_path = os.path.join('ckpt/', f'{params.RUNINFO.dataset_name}_model_best.pt')

    if os.path.exists(save_path):
        print(f"Loading best model from {save_path}")
        model.load_state_dict(torch.load(save_path))
        best_iter = _evaluate_model(params, model, test_loader, device, epoch=-1)
    else:
        best_epoch, best_val_metrics = _train_model(
            params, model, optimizer, train_loader, val_loader, device, output_f, save_path
        )
        print(f"Best validation at epoch {best_epoch:03d}: {best_val_metrics}", file=output_f)
        output_f.flush()
        model.load_state_dict(torch.load(save_path))
        best_iter = _evaluate_model(params, model, test_loader, device, epoch=best_epoch)

    _print_best_performance(params, best_iter, output_f)
    output_f.flush()
    return best_iter


def _collect_predictions(model, data_loader, device):
    preds, targets = [], []
    for batch in data_loader:
        batch = batch.to(device)
        pred, target, _, _, _ = model.test_predict_graph(batch)
        preds.append(pred)
        targets.append(target)
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def _is_better(params, current_metrics, best_score):
    if params.MODELINFO.decoder_type == 'regression':
        score = current_metrics["RMSE"]
        return score < best_score, score
    if params.MODELINFO.decoder_type == "binary":
        score = current_metrics["AUROC"]
        return score > best_score, score
    if params.MODELINFO.decoder_type == "multiclass":
        score = current_metrics["Accuracy"]
        return score > best_score, score
    raise ValueError("Unsupported decoder type: {}".format(params.MODELINFO.decoder_type))


def _evaluate_model(params, model, data_loader, device, epoch):
    """Run inference on a data split with a loaded checkpoint."""
    model.eval()
    pred, target = _collect_predictions(model, data_loader, device)

    if params.MODELINFO.decoder_type == 'regression':
        m = compute_regression_metrics(pred, target)
        return (epoch, pred, m["RMSE"], m["MAE"], m["Pearson"], m["R2"])
    elif params.MODELINFO.decoder_type == "binary":
        m = compute_binary_classification_metrics(pred, target)
        pred_label = torch.sigmoid(pred)
        return (epoch, pred_label, m["Accuracy"], m["F1"], m["AUROC"], m["AUPRC"])
    elif params.MODELINFO.decoder_type == "multiclass":
        m = compute_multiclass_classification_metrics(pred, target)
        pred_label = torch.argmax(pred, dim=1)
        return (epoch, pred_label, m["Accuracy"], m["MacroF1"], m["MicroF1"], m["WeightedF1"])


def _train_model(params, model, optimizer, train_loader, val_loader, device, output_f, save_path):
    """Run the full training loop and select checkpoint by validation metric."""
    decoder_type = params.MODELINFO.decoder_type

    if decoder_type == 'regression':
        best_score = np.inf
    else:
        best_score = -np.inf
    best_epoch = -1
    best_val_metrics = None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in tqdm(range(params.RUNPARAMS.epochs)):
        epoch_loss = 0.0
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            loss = model.train_graph(batch, penalize_fn=params.RUNPARAMS.penalize_false_negative)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        val_pred, val_target = _collect_predictions(model, val_loader, device)
        torch.cuda.empty_cache()

        if decoder_type == 'regression':
            m = compute_regression_metrics(val_pred, val_target)
            is_better, score = _is_better(params, m, best_score)
            if is_better:
                best_score = score
                best_epoch = epoch + 1
                best_val_metrics = m
                torch.save(model.state_dict(), save_path)
            print(
                f"Epoch: {epoch+1:03d}, Train Loss: {avg_loss:.4f}, Val RMSE: {m['RMSE']:.4f}, MAE: {m['MAE']:.4f}, "
                f"Pearson Coeff (Rp): {m['Pearson']:.4f}, R2: {m['R2']:.4f}",
                file=output_f,
            )
            output_f.flush()
            if np.isnan(avg_loss) and np.isnan(m["RMSE"]):
                logging.warning("Both train loss and val RMSE are NaN - gradient overflowed, stopping training")
                break

        elif decoder_type == "binary":
            m = compute_binary_classification_metrics(val_pred, val_target)
            is_better, score = _is_better(params, m, best_score)
            if is_better:
                best_score = score
                best_epoch = epoch + 1
                best_val_metrics = m
                torch.save(model.state_dict(), save_path)
            print(
                f"Epoch: {epoch+1:03d}, Train Loss: {avg_loss:.4f}, Val acc: {m['Accuracy']:.4f}, f1: {m['F1']:.4f}, "
                f"auroc: {m['AUROC']:.4f}, auprc: {m['AUPRC']:.4f}",
                file=output_f,
            )
            output_f.flush()

        elif decoder_type == "multiclass":
            m = compute_multiclass_classification_metrics(val_pred, val_target)
            is_better, score = _is_better(params, m, best_score)
            if is_better:
                best_score = score
                best_epoch = epoch + 1
                best_val_metrics = m
                torch.save(model.state_dict(), save_path)
            print(
                f"Epoch: {epoch+1:03d}, Train Loss: {avg_loss:.4f}, Val acc: {m['Accuracy']:.4f}, macro_f1: {m['MacroF1']:.4f}, "
                f"micro_f1: {m['MicroF1']:.4f}, weighted_f1: {m['WeightedF1']:.4f}",
                file=output_f,
            )
            output_f.flush()

    if best_epoch < 0:
        raise RuntimeError("No valid checkpoint selected from validation set.")
    return best_epoch, best_val_metrics


def _print_best_performance(params, best_iter, output_f):
    if params.MODELINFO.decoder_type == 'regression':
        print(
            f"Performances at best: Epoch {best_iter[0]:d} - Test RMSE = {best_iter[2]:.4f}, MAE = {best_iter[3]:.4f}, "
            f"Pearson Coeff (Rp) = {best_iter[4]:.4f}, R\u00b2 = {best_iter[5]:.4f}",
            file=output_f,
        )
    elif params.MODELINFO.decoder_type == "binary":
        print(
            f"Performances at best: Epoch {best_iter[0]:d} - Test acc = {best_iter[2]:.4f}, f1 = {best_iter[3]:.4f}, "
            f"auroc = {best_iter[4]:.4f}, auprc = {best_iter[5]:.4f}",
            file=output_f,
        )
    elif params.MODELINFO.decoder_type == "multiclass":
        print(
            f"Performances at best: Epoch {best_iter[0]:d} - Test acc = {best_iter[2]:.4f}, macro_f1 = {best_iter[3]:.4f}, "
            f"micro_f1 = {best_iter[4]:.4f}, weighted_f1 = {best_iter[5]:.4f}",
            file=output_f,
        )


def save_results(params, data, best_iter, model, output_f):
    preds = best_iter[1].flatten()
    targets = model.ground_truth(data, mode="test").float().flatten()

    if model.link_from == "target" and model.link_to == "compound":
        protein_nodes = model.inspect(data, mode="test")[0, :]
        compound_nodes = model.inspect(data, mode="test")[1, :]
    elif model.link_from == "compound" and model.link_to == "target":
        compound_nodes = model.inspect(data, mode="test")[0, :]
        protein_nodes = model.inspect(data, mode="test")[1, :]

    compound_nodes = np.array(data["compound"]['names'])[compound_nodes.tolist()]
    protein_nodes = np.array(data["target"]['names'])[protein_nodes.tolist()]

    if output_f != sys.stdout:
        pd.DataFrame({
            "Predictions": preds.cpu(),
            "Ground Truths": targets.cpu(),
            "Compound": compound_nodes,
            "Target": protein_nodes,
        }).to_csv(output_f.name.replace(".log", ".csv"))


def main(config_file, run_name_suffix=None):
    params = parse_params(config_file=config_file)
    run_name = params.RUNINFO.run_name
    log_name = f'{run_name}_{run_name_suffix}' if run_name_suffix else run_name
    output_f = load_config(params, run_name=log_name)

    set_seed(params.RUNPARAMS.seed)

    dataset, data = load_dataset(params)
    model = init_model(params, data, dataset.get_metadata(use_dtis=params.MODELINFO.use_dtis))
    best_iter = run_pipeline(params, data, model, output_f)
    save_results(params, data, best_iter, model, output_f)

    return best_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="InterMS Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_file", type=str, help="Path to config file", default="configs/pdbbind2020_config.ini")
    args, _ = parser.parse_known_args()
    main(args.config_file)
