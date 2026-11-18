import logging
import os
import random
import sys
from statistics import fmean

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.utils import parse_params
from src.chemical_properties import compound_dict, parallel_compute_compound_similarity, parallel_compute_compound_similarity_for_large
from src.data import (
    HeteroData,
    PDADataset,
    dataframe_raw_vectorization_with_numpy,
    parallel_ppi_processing,
)
from src.external import exclude_nonhuman_proteins
from src.biological_properties import protein_init
from src.chemical_properties import ligand_init
"""
Class adapted from the PDADataset class to load PDB Bind data
TODO: Refactor PDA Dataset to avoid such copy/paste!
"""

class PDBBindDataset(PDADataset):
    def __init__(self, settings, dataset_name="pdbbind2020"):
        self.dataset_name = dataset_name
        super().__init__(settings)
    
    @property
    def processed_file_names(self):
        return super().processed_file_names.replace("pda-data", "{}-data".format(self.dataset_name))

    def get(self):
        if os.path.isfile(os.path.join(self.data_folder, self.processed_file_names)):
            print(
                "PDBBind Dataset: Loading data from {} ...".format(
                    os.path.join(self.data_folder, self.processed_file_names)
                )
            )
            self.data = self.load_data()
        else:
            print("PDBBind Dataset: Building data...")
            self.data = self.build_process()
        return self.data

# BindingDB Dataset
class BindingDBDataset(PDADataset):
    def __init__(self, settings, dataset_name="bindingdb"):
        self.dataset_name = dataset_name
        super().__init__(settings)
    
    @property
    def processed_file_names(self):
        return super().processed_file_names.replace("pda-data", "{}-data".format(self.dataset_name))

    def get(self):
        if os.path.isfile(os.path.join(self.data_folder, self.processed_file_names)):
            print(
                "bindingdb Dataset: Loading data from {} ...".format(
                    os.path.join(self.data_folder, self.processed_file_names)
                )
            )
            self.data = self.load_data()
        else:
            print("bindingdb Dataset: Building data...")
            self.data = self.build_process()
        return self.data
    
# BioSNAP 
class BioSNAPDataset(PDADataset):
    def __init__(self, settings, dataset_name="biosnap"):
        # Specify which part of the PDB Bind dataset to build 
        self.dataset_name = dataset_name
        super().__init__(settings)
    
    @property
    def processed_file_names(self):
        return super().processed_file_names.replace("pda-data", "{}-data".format(self.dataset_name))

    def get(self):
        if os.path.isfile(os.path.join(self.data_folder, self.processed_file_names)):
            print(
                "BioSNAP Dataset: Loading data from {} ...".format(
                    os.path.join(self.data_folder, self.processed_file_names)
                )
            )
            self.data = torch.load(os.path.join(self.data_folder, self.processed_file_names))
        else:
            print("BioSNAP Dataset: Building data...")
            self.data = self.build_process()
        return self.data


# function 
class FunctionDataset(PDADataset):
    def __init__(self, settings, dataset_name="function"):
        # Specify which part of the PDB Bind dataset to build 
        self.dataset_name = dataset_name
        super().__init__(settings)
    
    @property
    def processed_file_names(self):
        return super().processed_file_names.replace("pda-data", "{}-data".format(self.dataset_name))

    def read_data(self):
        try:
            ppi_df = pd.read_csv(
                self.settings.INPUT_FILES.ppi_database,
                header=None,
                names=["Prot1", "Prot2", "Score"],
                sep=' '
            )
            pdbbind_all_index = pd.read_csv(
                os.path.join(self.data_folder, 'full.csv')
            )
        except Exception:
            raise ValueError(
                "BioSNAP data not found in {} - please fix your folder containing biosnap data accordingly".format(
                    self.data_folder
                )
            )
        pdbbind_all_index = pdbbind_all_index
        self.targets = pdbbind_all_index["UniProt ID"].unique()
        self.targets = exclude_nonhuman_proteins(
            self.targets, human_path=self.settings.INPUT_FILES.human_proteins, json_path=os.path.join(self.data_folder, "biosnap_human_proteins.json") # 1095-> 1193
        )

        mapping_file = pd.read_csv(self.settings.INPUT_FILES.mapping_file)
        mapping_file = mapping_file[mapping_file.Accession.isin(self.targets)]

        pdbbind_all_index = pdbbind_all_index[pdbbind_all_index["UniProt ID"].isin(self.targets)]
        self.compounds = pdbbind_all_index["SMILES"].unique().tolist()

        pdbbind_all_index = pdbbind_all_index[pdbbind_all_index["SMILES"].isin(self.compounds)]

        self.pdbbind_all_index = pdbbind_all_index.reset_index(drop=True)

        return pdbbind_all_index, ppi_df, mapping_file

    def build_graph(self):
        pdbbind_all_index, ppi_df, mapping_file = self.read_data()

        print(
            "Initiating a graph containing {} compounds and {} targets...".format(
                len(self.compounds), len(self.targets)
            )
        )

        # Protein features
        target_features = np.empty(
            (
                len(self.targets),
                1 # placeholder
            ),
            dtype=np.float32,
        )

        protein_path = os.path.join(self.data_folder, f'{self.dataset_name}_protein_graphs.pt')
        if os.path.exists(protein_path):
            print("Loading cached protein graphs...")
            protein_smallgraph_dict = torch.load(protein_path)
            assert len(protein_smallgraph_dict) == len(self.targets), "The number of protein graph dict does not match the number of nodes in the large graph."
        else:
            print("Generating protein graphs...")
            protein_seqs = []
            for idx, target in enumerate(tqdm(self.targets, file=sys.stdout, desc="Exacting Protein Seq")):
                seq = pdbbind_all_index.loc[pdbbind_all_index['UniProt ID'] == target, 'Target']
                if not seq.empty:
                    protein_seqs.append(seq.iloc[0])

            assert len(protein_seqs) == len(self.targets), "There are protein sequences that were not successfully found."
            protein_smallgraph_dict = protein_init(protein_seqs)
            assert len(protein_smallgraph_dict) == len(self.targets), "There are protein sequences that were not successfully constructed."
            torch.save(protein_smallgraph_dict, os.path.join(self.data_folder, f'{self.dataset_name}_protein_graphs.pt'))

        # compound features
        compound_features = np.empty(
            (
                len(self.compounds),
                1, # placeholder
            ),
            dtype=np.float32,
        )
        ligand_path = os.path.join(self.data_folder, f'{self.dataset_name}_ligand_graphs.pt')
        if os.path.exists(ligand_path):
            print("Loading cached ligand graphs...")
            protein_smallgraph_dict = torch.load(ligand_path)
            assert len(protein_smallgraph_dict) == len(self.compounds), "The number of ligand graph dict does not match the number of nodes in the large graph."
        else:
            print("Generating ligand graphs...")
            ligand_smallgraph_dict = ligand_init(self.compounds)
            assert len(ligand_smallgraph_dict) == len(self.compounds), "There are smiles sequences that were not successfully constructed."
            torch.save(ligand_smallgraph_dict, os.path.join(self.data_folder, f'{self.dataset_name}_ligand_graphs.pt'))

        # Edges for compound-targets interactions
        edges_inter_from_to = []
        # edges_inter_features = []
        edges_inter_y = []
        edges_inter_train_idx, edges_inter_val_idx, edges_inter_test_idx = [], [], []

        # Create a dict of experimental label
        edge_inter_dict_train = {prot: {compound: [] for compound in self.compounds} for prot in self.targets}
        edge_inter_dict_val = {prot: {compound: [] for compound in self.compounds} for prot in self.targets}
        edge_inter_dict_test = {prot: {compound: [] for compound in self.compounds} for prot in self.targets}
        # Get label info
        for target_uniprot_id, compound_smiles_name, label, source in tqdm(
            dataframe_raw_vectorization_with_numpy(
                pdbbind_all_index, keys=["UniProt ID", "SMILES", "Label", "Source"]
            ),
            file=sys.stdout,
            desc="PLI Edges Collecting results",
        ):
            if compound_smiles_name in self.compounds and target_uniprot_id in self.targets:
                if source == 'train':
                    edge_inter_dict_train[target_uniprot_id][compound_smiles_name].append(label)
                elif source == 'val':
                    edge_inter_dict_val[target_uniprot_id][compound_smiles_name].append(label)
                elif source == 'test':
                    edge_inter_dict_test[target_uniprot_id][compound_smiles_name].append(label)
            else:
                logging.warning(
                    "BioSNAPDataset: label recorded on unknown drug or target (compound {} - target {})".format(
                        compound_smiles_name, target_uniprot_id
                    )
                )

        idx_count = 0 # converted to a sparse matrix
        for uniprot_id in self.targets:
            for compound_smiles_name in self.compounds:
                #  Used for train set
                if edge_inter_dict_train[uniprot_id][compound_smiles_name]:  # list is not empty
                    edges_inter_from_to.append([compound_smiles_name, uniprot_id])
                    edges_inter_train_idx.append(idx_count)
                    idx_count += 1
                    # Pick randomly one of the label and fill in edges
                    edges_inter_y.append([random.choice(edge_inter_dict_train[uniprot_id][compound_smiles_name]) + 1])

                #  Used for val set
                if edge_inter_dict_val[uniprot_id][compound_smiles_name]:  # list is not empty
                    edges_inter_from_to.append([compound_smiles_name, uniprot_id])
                    edges_inter_val_idx.append(idx_count)
                    idx_count += 1
                    # Pick randomly one of the label and fill in edges
                    edges_inter_y.append([random.choice(edge_inter_dict_val[uniprot_id][compound_smiles_name]) + 1])

                #  Used for test set
                if edge_inter_dict_test[uniprot_id][compound_smiles_name]:  # list is not empty
                    edges_inter_from_to.append([compound_smiles_name, uniprot_id])
                    edges_inter_test_idx.append(idx_count)
                    idx_count += 1
                    # Pick randomly one of the label and fill in edges
                    edges_inter_y.append(
                        [random.choice(edge_inter_dict_test[uniprot_id][compound_smiles_name]) + 1])

        # Edges for protein-protein interactions
        edges_ppi_from_to, edges_ppi_features = parallel_ppi_processing(ppi_df=ppi_df, mapping_file=mapping_file, targets=self.targets, 
                                                                        ppi_threshold=self.settings.PROTEIN_FEATURES.ppi_comb_score_threshold, n_procs=16)

        # Edges for compound-compound interaction
        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:
            similarity_threshold = self.get_compound_similarity_threshold()
  
            edges_drugs_from_to, edges_drugs_features = parallel_compute_compound_similarity_for_large(
                compounds=self.compounds,
                similarity_threshold=similarity_threshold,
                use_compound_similarities=self.settings.COMPOUND_FEATURES.compound_similarities,
                n_procs=96,        # depends on the number of CPU cores
                chunk_size=500000  # depends on the actual data size
            )
            
        # Format data into numpy arrays
        target_features = np.array(target_features)
        compound_features = np.array(compound_features)
        edges_inter_from_to = np.array(
            [
                [self.compounds.index(edge_from) for edge_from, _ in edges_inter_from_to],
                [self.targets.index(edge_to) for _, edge_to in edges_inter_from_to],
            ]
        )
        # edges_inter_features = np.array(edges_inter_features)
        
        edges_ppi_from_to = np.array(
            [
                [self.targets.index(edge_from) for edge_from, _ in edges_ppi_from_to],
                [self.targets.index(edge_to) for _, edge_to in edges_ppi_from_to],
            ]
        )
        edges_ppi_features = np.array(edges_ppi_features)

        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:
            # Draw edges unless 'None' compound similarities
            edges_drugs_from_to = edges_drugs_from_to
            edges_drugs_features = edges_drugs_features

        # Create the graph
        data = HeteroData()

        data["target"].x = torch.Tensor(target_features)
        data["target"].names = self.targets
        data["compound"].x = torch.Tensor(compound_features)
        data["compound"].names = self.compounds
        data["compound"].smiles = self.compounds

        if self.settings.PROTEIN_FEATURES.protein_protein_interactions:
            data["target", "interaction", "target"].edge_index = torch.LongTensor(edges_ppi_from_to)
            data["target", "interaction", "target"].edge_attr = torch.Tensor(edges_ppi_features)
        else:
            data["target", "interaction", "target"].edge_index = torch.ones((2, 0)).long()
            data["target", "interaction", "target"].edge_attr = torch.ones((0, 1)).float()

        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:
            data["compound", "similarity", "compound"].edge_index = torch.LongTensor(edges_drugs_from_to)
            data["compound", "similarity", "compound"].edge_attr = torch.Tensor(edges_drugs_features)
        else:
            data["compound", "similarity", "compound"].edge_index = torch.ones((2, 0)).long()
            data["compound", "similarity", "compound"].edge_attr = torch.ones((0, 1)).float()

        # Set label to predict as pAct for protein-drug interaction
        data["compound", "interaction", "target"].edge_index = torch.ones(
            (2, 0)
        ).long()  # Create empty edge index - for compatibility
        data["compound", "interaction", "target"].edge_attr = torch.ones(
            (0, 1)
        ).float()  # Create empty edge attr - for compatibility
        data["compound", "interaction", "target"].y_edge_index = torch.LongTensor(edges_inter_from_to)
        data["compound", "interaction", "target"].y = torch.Tensor(edges_inter_y)
        data["compound", "interaction", "target"].train_idx = torch.LongTensor(edges_inter_train_idx)
        data["compound", "interaction", "target"].val_idx = torch.LongTensor(edges_inter_val_idx)
        data["compound", "interaction", "target"].test_idx = torch.LongTensor(edges_inter_test_idx)

        data = self.transform_graph(data)

        print("Graph initialized!\n{}\n\n".format(data))
        return data

    def get(self):
        if os.path.isfile(os.path.join(self.data_folder, self.processed_file_names)):
            print(
                "Function Dataset: Loading data from {} ...".format(
                    os.path.join(self.data_folder, self.processed_file_names)
                )
            )
            self.data = torch.load(os.path.join(self.data_folder, self.processed_file_names))
        else:
            print("Function Dataset: Building data...")
            self.data = self.build_process()
        return self.data


if __name__ == "__main__":

    # test pdbbind dataset
    params = parse_params(config_file='configs/pdbbind2020_config.ini')
    dataset = PDBBindDataset(settings=params, dataset_name=params.RUNINFO.dataset_name)

    # test bindingdb dataset
    # params = parse_params(config_file='configs/bindingdb_config.ini')
    # dataset = BindingDBDataset(settings=params, dataset_name=params.RUNINFO.dataset_name)
    
    # test biosnap dataset
    # params = parse_params(config_file='configs/biosnap_config.ini')
    # dataset = BioSNAPDataset(settings=params, dataset_name="biosnap")
    
    # test function dataset
    # params = parse_params(config_file='configs/function_config.ini')
    # dataset = FunctionDataset(settings=params, dataset_name="function")

    data = dataset.get()
    print(data)
    # dataset.export_data(data, uniprot2seq=True, protein_graph=f'{params.RUNINFO.dataset_name}_protein_graphs.pt')

