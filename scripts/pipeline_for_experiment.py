import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import load_config, parse_params
from src.metrics import (
    compute_regression_metrics,
    compute_binary_classification_metrics,
    compute_multiclass_classification_metrics,
)
from scripts.pipeline import (
    set_seed,
    load_dataset,
    init_model,
    build_data_loaders,
    save_results,
    _print_best_performance,
    _is_better,
    _collect_predictions,
)


def run_experiment_pipeline(params, data, model, output_f):
    """Training + evaluation pipeline with interpretability analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() and not params.RUNINFO.force_cpu else "cpu")
    model = model.to(device)

    num_params = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_params:d}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.RUNPARAMS.learning_rate, weight_decay=1e-4)

    train_loader, val_loader, test_loader = build_data_loaders(params, data, model)

    decoder_type = params.MODELINFO.decoder_type
    save_path = os.path.join('ckpt/', f'{params.RUNINFO.dataset_name}_model_best.pt')

    if os.path.exists(save_path):
        print(f"Loading best model from {save_path}")
        model.load_state_dict(torch.load(save_path))
        best_iter = _evaluate_with_explain(params, model, test_loader, device, epoch=-1)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        best_epoch, best_val_metrics = _train_experiment(
            params, model, optimizer, train_loader, val_loader, device, output_f, save_path, data
        )
        print(f"Best validation at epoch {best_epoch:03d}: {best_val_metrics}", file=output_f)
        output_f.flush()
        model.load_state_dict(torch.load(save_path))
        best_iter = _evaluate_with_explain(params, model, test_loader, device, epoch=best_epoch)

    _print_best_performance(params, best_iter, output_f)
    output_f.flush()
    return best_iter


def _evaluate_with_explain(params, model, test_loader, device, epoch):
    """Evaluate + run interpretability on test set."""
    model.eval()
    test_preds, test_targets = [], []

    for i, batch in enumerate(test_loader):
        batch = batch.to(device)
        pred, target, _, _, _ = model.test_predict_graph(batch)
        model.explain_local(
            batch,
            save_dir=f'outputs/explain_results/{params.RUNINFO.dataset_name}/batch_{i+1}',
            explain_protein=True,
            explain_ligand=True,
        )
        test_preds.append(pred)
        test_targets.append(target)

    test_pred = torch.cat(test_preds, dim=0)
    test_target = torch.cat(test_targets, dim=0)

    if params.MODELINFO.decoder_type == 'regression':
        m = compute_regression_metrics(test_pred, test_target)
        return (epoch, test_pred, m["RMSE"], m["MAE"], m["Pearson"], m["R2"])
    elif params.MODELINFO.decoder_type == "binary":
        m = compute_binary_classification_metrics(test_pred, test_target)
        pred_label = torch.sigmoid(test_pred)
        return (epoch, pred_label, m["Accuracy"], m["F1"], m["AUROC"], m["AUPRC"])
    elif params.MODELINFO.decoder_type == "multiclass":
        m = compute_multiclass_classification_metrics(test_pred, test_target)
        pred_label = torch.argmax(test_pred, dim=1)
        return (epoch, pred_label, m["Accuracy"], m["MacroF1"], m["MicroF1"], m["WeightedF1"])


def _train_experiment(params, model, optimizer, train_loader, val_loader, device, output_f, save_path, data):
    """Training loop for experiment pipeline, selecting checkpoint by validation metrics."""
    decoder_type = params.MODELINFO.decoder_type

    if decoder_type == 'regression':
        best_score = np.inf
    else:
        best_score = -np.inf
    best_epoch = -1
    best_val_metrics = None

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
            pred_label = (torch.sigmoid(val_pred) >= 0.5).int()
            _save_step_results(params, data, pred_label, model, epoch + 1, mode="val")
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


def _save_step_results(params, data, preds, model, epoch, mode="val"):
    """Save per-epoch prediction results for binary classification experiments."""
    preds = preds.flatten()
    targets = model.ground_truth(data, mode=mode).float().flatten()

    if model.link_from == "target" and model.link_to == "compound":
        protein_nodes = model.inspect(data, mode=mode)[0, :]
        compound_nodes = model.inspect(data, mode=mode)[1, :]
    elif model.link_from == "compound" and model.link_to == "target":
        compound_nodes = model.inspect(data, mode=mode)[0, :]
        protein_nodes = model.inspect(data, mode=mode)[1, :]

    compound_nodes = np.array(data["compound"]['names'])[compound_nodes.tolist()]
    protein_nodes = np.array(data["target"]['names'])[protein_nodes.tolist()]

    out_dir = f"outputs/experiment_steps/{params.RUNINFO.dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        "Predictions": preds.cpu(),
        "Ground Truths": targets.cpu(),
        "Compound": compound_nodes,
        "Target": protein_nodes,
    }).to_csv(os.path.join(out_dir, f"results_epoch{epoch}.csv"))


def main(config_file):
    params = parse_params(config_file=config_file)
    run_name = params.RUNINFO.run_name
    output_f = load_config(params, run_name=run_name)

    set_seed(params.RUNPARAMS.seed)

    dataset, data = load_dataset(params)
    model = init_model(params, data, dataset.get_metadata(use_dtis=params.MODELINFO.use_dtis))
    best_iter = run_experiment_pipeline(params, data, model, output_f)
    save_results(params, data, best_iter, model, output_f)

    return best_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="InterMS Experiment Pipeline (with interpretability)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_file", type=str, help="Path to config file", default="configs/biosnap_config.ini")
    args, _ = parser.parse_known_args()
    main(args.config_file)
