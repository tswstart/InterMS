import json
import copy
import logging
import os
import random
import sys
from statistics import median

import numpy as np
import pandas as pd
import torch

import torch_geometric.transforms as T
from rdkit import Chem
from torch.distributions.uniform import Uniform
from torch_geometric.data import Dataset, HeteroData
from tqdm.auto import tqdm
from statistics import fmean

from src.biological_properties import (
    parallel_ppi_processing,
    protein_init
)
from src.chemical_properties import (
    compute_new_compound_similarity,
    compound_dict,
    ligand_init,
    parallel_compute_compound_similarity, parallel_compute_compound_similarity_for_large
)
from src.external import download_uniprot_fasta, retrieve_protein_classes, exclude_nonhuman_proteins
from src.utils import dataframe_raw_vectorization_with_numpy

# basic logging config
if not os.path.isdir("outputs"):
    os.makedirs("outputs")
logging.basicConfig(filename="data/construct_data.log", level=logging.DEBUG)


class PDADataset(Dataset):
    def __init__(self, settings):
        self.data_folder = settings.RUNINFO.data_dir
        self.preprocessed_data_path = os.path.join(self.data_folder, "preprocessed.json")
        self.settings = settings  # Contains all the customized setting
        super().__init__("./", transform=None, pre_filter=None, pre_transform=None)

    # Create common and correct name to save the data, from the defined setting
    @property
    def processed_file_names(self):
        base = "pda-data"

        if self.settings.INPUT_FILES.average_screenings:
            # In case several screenings are available for the same PLI
            base = base + "_Averaged"

        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:
            comp_sim_value = (
                str(self.get_compound_similarity_threshold())
                if self.get_compound_similarity_threshold() != 0.0
                else ""
            )
            if self.settings.COMPOUND_FEATURES.compound_similarities:
                base = base + "_CompSimilarities{}".format(comp_sim_value)
            else:
                base = base + "_NoCompSimilarities"
        else:
            base = base + "_NoCompLinks"

        if self.settings.PROTEIN_FEATURES.protein_protein_interactions:
            base = base + "_PPI"
            if self.settings.PROTEIN_FEATURES.ppi_comb_score_threshold is not None:
                base = base + "_Score{}".format(self.settings.PROTEIN_FEATURES.ppi_comb_score_threshold)    
        else:
            base = base + "_noPPI"

        # Add PPI database used in the save name
        return base + ".pt"

    def load_data(self):
        self.data = torch.load(os.path.join(self.data_folder, self.processed_file_names))
        self.targets = self.data["target"]["names"]
        self.compounds = self.data["compound"]["names"]
        return self.data

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
                "pdbbind data not found in {} - please fix your folder containing PDB data accordingly".format(
                    self.data_folder
                )
            )

        pdbbind_all_index = pdbbind_all_index
        # clear none uniprot id
        self.targets = pdbbind_all_index[pdbbind_all_index["UniProt ID"].notna()]["UniProt ID"].unique().tolist()

        self.targets = exclude_nonhuman_proteins(
            self.targets, human_path=self.settings.INPUT_FILES.human_proteins, json_path=os.path.join(self.data_folder, "human_proteins.json") # 1095-> 1193
        )

        mapping_file = pd.read_csv(self.settings.INPUT_FILES.mapping_file)
        mapping_file = mapping_file[mapping_file.Accession.isin(self.targets)]

        pdbbind_all_index = pdbbind_all_index[pdbbind_all_index["UniProt ID"].isin(self.targets)]
        self.compounds = pdbbind_all_index["SMILES"].unique().tolist()

        # Save correct SMILES and exclude miscallaneous ones (which were not suitable to calculate compound features)
        smiles_eq_path = os.path.join(self.data_folder, "smiles_mapping.json")
        if os.path.isfile(smiles_eq_path):
            with open(smiles_eq_path) as json_file:
                smiles_to_equivalent_save = json.load(json_file)
                smiles_to_equivalent, invalid_smiles = (
                    smiles_to_equivalent_save["smiles_dict"],
                    smiles_to_equivalent_save["invalid_smiles"],
                )
        else:
            smiles_to_equivalent, invalid_smiles = compound_dict(pdbbind_all_index)
            smiles_to_equivalent_save = {"smiles_dict": smiles_to_equivalent, "invalid_smiles": invalid_smiles}
            with open(smiles_eq_path, "w+") as json_file:
                json.dump(smiles_to_equivalent_save, json_file)
        self.compounds = np.unique(
            [k for k, v in smiles_to_equivalent.items() if k in pdbbind_all_index["SMILES"].tolist()]
        ).tolist()
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
                1
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
                seq = pdbbind_all_index.loc[pdbbind_all_index['UniProt ID'] == target, 'Target Sequence']
                if not seq.empty:
                    protein_seqs.append(seq.iloc[0])
                else:
                    print(f"Warning: No sequence found for target {target}")

            assert len(protein_seqs) == len(self.targets), "There are protein sequences that were not successfully found."
            protein_smallgraph_dict = protein_init(protein_seqs)
            assert len(protein_smallgraph_dict) == len(self.targets), "There are protein sequences that were not successfully constructed."
            torch.save(protein_smallgraph_dict, os.path.join(self.data_folder, f'{self.dataset_name}_protein_graphs.pt'))

        # compound features
        compound_features = np.empty(
            (
                len(self.compounds),
                1,
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
                    "Dataset: label recorded on unknown drug or target (compound {} - target {})".format(
                        compound_smiles_name, target_uniprot_id
                    )
                )
        
        idx_count = 0 #
        for uniprot_id in self.targets:
            for compound_smiles_name in self.compounds:
                #  Used for train set
                if edge_inter_dict_train[uniprot_id][compound_smiles_name]:  # list is not empty
                    edges_inter_from_to.append([compound_smiles_name, uniprot_id])
                    edges_inter_train_idx.append(idx_count)
                    idx_count += 1
                    if self.settings.INPUT_FILES.average_screenings:
                        # Average and fill in edges
                        edges_inter_y.append([fmean(edge_inter_dict_train[uniprot_id][compound_smiles_name])])
                    else:
                        # Pick randomly one of the pAct and fill in edges
                        edges_inter_y.append([random.choice(edge_inter_dict_train[uniprot_id][compound_smiles_name])])

                #  Used for val set
                if edge_inter_dict_val[uniprot_id][compound_smiles_name]:  # list is not empty
                    edges_inter_from_to.append([compound_smiles_name, uniprot_id])
                    edges_inter_val_idx.append(idx_count)
                    idx_count += 1
                    if self.settings.INPUT_FILES.average_screenings:
                        # Average and fill in edges
                        edges_inter_y.append([fmean(edge_inter_dict_val[uniprot_id][compound_smiles_name])])
                    else:
                        # Pick randomly one of the pAct and fill in edges
                        edges_inter_y.append([random.choice(edge_inter_dict_val[uniprot_id][compound_smiles_name])])

                #  Used for test set
                if edge_inter_dict_test[uniprot_id][compound_smiles_name]:  # list is not empty
                    edges_inter_from_to.append([compound_smiles_name, uniprot_id])
                    edges_inter_test_idx.append(idx_count)
                    idx_count += 1
                    if self.settings.INPUT_FILES.average_screenings:
                        # Average and fill in edges
                        edges_inter_y.append([fmean(edge_inter_dict_test[uniprot_id][compound_smiles_name])])
                    else:
                        # Pick randomly one of the pAct and fill in edges
                        edges_inter_y.append(
                            [random.choice(edge_inter_dict_test[uniprot_id][compound_smiles_name])]
                        )
            
        # Edges for protein-protein interactions
        edges_ppi_from_to, edges_ppi_features = parallel_ppi_processing(ppi_df=ppi_df, mapping_file=mapping_file, targets=self.targets, 
                                                                        ppi_threshold=self.settings.PROTEIN_FEATURES.ppi_comb_score_threshold, n_procs=16)

        # Edges for compound-compound interaction
        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:
            similarity_threshold = self.get_compound_similarity_threshold()
  
            edges_drugs_from_to, edges_drugs_features = parallel_compute_compound_similarity(
                compounds=self.compounds,
                similarity_threshold=similarity_threshold,
                use_compound_similarities=self.settings.COMPOUND_FEATURES.compound_similarities,
                n_procs=32,
                chunk_size=1000000
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
            edges_drugs_from_to = np.array(
                [
                    [self.compounds.index(edge_from) for edge_from, _ in edges_drugs_from_to],
                    [self.compounds.index(edge_to) for _, edge_to in edges_drugs_from_to],
                ]
            )
            edges_drugs_features = np.array(edges_drugs_features)

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
        

    def add_target_nodes(self, node_names):
        ppi_df = pd.read_csv(
            self.settings.INPUT_FILES.ppi_database,
            header=None,
            names=["Prot1", "Prot2", "Score"],
            sep=' '
        )

        valid_node_names = []
        for node_name in node_names:
            if node_name in self.data["target"].names:
                logging.warning("PDA Dataset: Trying to duplicate node {}".format(node_name))
            else:
                valid_node_names.append(node_name)

        if len(valid_node_names) == 0:
            return []

        mapping_file = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.mapping_file), delimiter="\t"
        )
        gene_expr_df = pd.read_csv(
            os.path.join(self.data_folder, self.settings.INPUT_FILES.gene_expr_database), delimiter="\t"
        )

        download_uniprot_fasta(valid_node_names, self.data_folder)

        temp_node_names, excluded_targets = self.check_ensembl_id(
            valid_node_names, mapping_file
        )  # Check if every target have a correct EnsemblGeneID, exclude otherwise
        if len(temp_node_names) == 0:
            logging.error("PDA Dataset: Target {} not found in mapping file".format(excluded_targets))
            return None
        valid_node_names, excluded_targets = self.check_proteins_validity(
            temp_node_names, mapping_file, gene_expr_df
        )  # Check if every target is successfully compiled, exclude otherwise
        if len(valid_node_names) == 0:
            logging.error("PDA Dataset: Could not compute Target {}'s features".format(excluded_targets))
            return None

        try:
            # Retrieve diverse protein classes from UniProt
            self.target_classes.append(retrieve_protein_classes([node_name for node_name in valid_node_names]))
        except Exception:
            pass

        target_features = np.empty(
            (
                len(valid_node_names),
                1,
            ),
            dtype=np.float32,
        )
        # target_esm_encodings = generate_esm_encodings(valid_node_names)
        # for idx, target in tqdm(enumerate(valid_node_names), file=sys.stdout, desc="Calculate protein features..."):
        #     target_ensembl = mapping_file[mapping_file.Accession == target].EnsemblGeneID.unique().tolist()
        #     target_features[idx, :] = get_protein_features(
        #         target,
        #         target_ensembl,
        #         gene_expr_df,
        #         target_esm_encodings,
        #         self.data_folder,
        #         khmer=self.settings.PROTEIN_FEATURES.khmer,
        #         k_khmer=self.settings.PROTEIN_FEATURES.khmer_size,
        #         sublocation=self.settings.PROTEIN_FEATURES.subcellular_location,
        #         gene_expr=self.settings.PROTEIN_FEATURES.gene_expression,
        #         esm_encoding=self.settings.PROTEIN_FEATURES.esm_encoding,
        #         gene_expr_tissue=self.settings.PROTEIN_FEATURES.gene_expression_tissue,
        #     )
        protein_path = os.path.join(self.data_folder, 'pdbbind_protein_graphs.pt')
        if os.path.exists(protein_path):
            original_graphs = torch.load(protein_path)
            new_graph_dict = ligand_init(valid_node_names)
            assert len(new_graph_dict) == len(valid_node_names), "There are smiles sequences that were not successfully constructed."
            
            max_index = max(original_graphs.keys()) if original_graphs else -1

            for idx, value in new_graph_dict.items():
                new_index = max_index + 1 + idx
                original_graphs[new_index] = value

            torch.save(original_graphs, os.path.join(self.data_folder, 'updated_pdbbind_protein_graphs.pt'))

        self.data["target"].x = torch.cat((self.data["target"].x, torch.from_numpy(target_features)), dim=0)
        self.data["target"].names = self.data["target"].names + valid_node_names
        self.targets = self.targets + valid_node_names
        assert len(self.targets) == len(np.unique(self.targets)), "PDA Dataset: Duplicated targets in dataset"

        # Extend protein-protein interaction network
        if self.settings.PROTEIN_FEATURES.protein_protein_interactions:
            edges_ppi_from_to = []
            edges_ppi_features = []

            edges_ppi_from_to, edges_ppi_features = parallel_ppi_processing(ppi_df=ppi_df, mapping_file=mapping_file, targets=self.targets, 
                                                                        ppi_threshold=self.settings.PROTEIN_FEATURES.ppi_comb_score_threshold, n_procs=16)

            self.data["target", "interaction", "target"].edge_index = torch.LongTensor(
                [
                    [self.targets.index(edge_from) for edge_from, _ in edges_ppi_from_to],
                    [self.targets.index(edge_to) for _, edge_to in edges_ppi_from_to],
                ]
            )
            self.data["target", "interaction", "target"].edge_attr = torch.Tensor(edges_ppi_features)

        return valid_node_names

    def add_compound_nodes(self, node_names, smiles_dict=None):
        valid_node_names = []
        for node_name in node_names:
            if node_name in self.data["compound"].names or node_name in self.data["compound"].smiles:
                logging.warning("PDA Dataset: Trying to duplicate node {}".format(node_name))
            else:
                try:
                    if smiles_dict is not None:
                        if node_name in smiles_dict:
                            Chem.MolFromSmiles(smiles_dict[node_name])
                            valid_node_names.append(smiles_dict[node_name])
                        else:
                            logging.warning("PDA Dataset: Trying to add node {} with no SMILES".format(node_name))
                            continue
                    else:
                        Chem.MolFromSmiles(node_name)
                        valid_node_names.append(node_name)
                except Exception:
                    logging.warning("PDA Dataset: Trying to add invalid SMILES {}".format(node_name))
                    continue

        # Computation of chemical features
        compound_features = np.empty(
            (
                len(valid_node_names),
                1,
            ),
            dtype=np.float32,
        )
        
        ligand_path = os.path.join(self.data_folder, 'pdbbind_ligand_graphs.pt')
        if os.path.exists(ligand_path):
            original_graphs = torch.load(ligand_path)

            new_graph_dict = ligand_init(valid_node_names)
            assert len(new_graph_dict) == len(valid_node_names), "There are smiles sequences that were not successfully constructed."
    
            max_index = max(original_graphs.keys()) if original_graphs else -1

            for idx, value in new_graph_dict.items():
                new_index = max_index + 1 + idx  
                original_graphs[new_index] = value 

            torch.save(original_graphs, os.path.join(self.data_folder, 'updated_pdbbind_ligand_graphs.pt'))

        self.data["compound"].x = torch.cat((self.data["compound"].x, torch.from_numpy(compound_features)), dim=0)
        self.compounds = self.compounds + valid_node_names
        assert len(self.compounds) == len(np.unique(self.compounds)), "PDA Dataset: Duplicated compounds in dataset"
        self.data["compound"].names = self.data["compound"].names + valid_node_names
        self.data["compound"].smiles = self.data["compound"].smiles + [comp for comp in valid_node_names]

        # For new nodes, add edges for compound-compound interaction
        if self.settings.COMPOUND_FEATURES.compound_similarities is not None:  # If none, compounds are not linked
            similarity_threshold = self.get_compound_similarity_threshold()

            edges_drugs_from_to = []
            edges_drugs_features = []

            # Edges new to all compounds (including new)    
            edges_drugs_from_to, edges_drugs_features = compute_new_compound_similarity(compounds_1=valid_node_names, 
                                compounds_2=self.data["compound"].smiles, similarity_threshold=similarity_threshold, 
                                use_compound_similarities=self.settings.COMPOUND_FEATURES.compound_similarities,
                                n_procs=32,
                                chunk_size=1000000
                            )

            # Draw edges unless 'None' compound similarities
            self.data["compound", "similarity", "compound"].edge_index = torch.LongTensor(
                [
                    self.data["compound", "similarity", "compound"].edge_index[0].tolist()
                    + [self.data["compound"].smiles.index(edge_from) for edge_from, _ in edges_drugs_from_to],
                    self.data["compound", "similarity", "compound"].edge_index[1].tolist()
                    + [self.data["compound"].smiles.index(edge_to) for _, edge_to in edges_drugs_from_to],
                ]
            )
            self.data["compound", "similarity", "compound"].edge_attr = torch.cat(
                (self.data["compound", "similarity", "compound"].edge_attr, torch.Tensor(edges_drugs_features)), dim=0
            )

        return valid_node_names

    def add_labeled_edges(self, compound_node_names, target_node_names, edge_labels):
        assert (
            len(compound_node_names) == len(target_node_names) == len(edge_labels)
        ), "PDA Dataset: Invalid number of edges"

        existing_edges = [(edge[0].item(), edge[1].item()) for edge in self.data["compound", "target"].y_edge_index.T]
        valid_edges = []
        valid_labels = []
        for compound, target, label in zip(compound_node_names, target_node_names, edge_labels):
            if compound in self.data["compound"].names and target in self.data["target"].names:
                compound_idx = self.data["compound"].names.index(compound)
                target_idx = self.data["target"].names.index(target)
                if (compound_idx, target_idx) not in existing_edges:
                    valid_edges.append([compound_idx, target_idx])
                    valid_labels.append([label])
                else:
                    logging.warning("PDA Dataset: Edge {}-{} not added".format(compound_idx, target_idx))
        if len(valid_edges) != 0:
            self.data["compound", "target"].y_edge_index = torch.cat(
                (self.data["compound", "target"].y_edge_index, torch.LongTensor(valid_edges).T), dim=1
            )
            self.data["compound", "target"].y = torch.cat(
                (self.data["compound", "target"].y, torch.Tensor(valid_labels)), dim=0
            )
            self.data["target", "compound"].y_edge_index = torch.cat(
                (
                    self.data["target", "compound"].y_edge_index,
                    torch.LongTensor([edge[::-1] for edge in valid_edges]).T,
                ),
                dim=1,
            )
            self.data["target", "compound"].y = torch.cat(
                (self.data["target", "compound"].y, torch.Tensor(valid_labels)), dim=0
            )

        return valid_edges

    def get_compound_similarity_threshold(self):
        if (
            self.settings.COMPOUND_FEATURES.compound_similarities_threshold >= 0
            and self.settings.COMPOUND_FEATURES.compound_similarities_threshold < 1
        ):
            return self.settings.COMPOUND_FEATURES.compound_similarities_threshold
        else:
            logging.warning(
                "PDA Dataset: Compound similarity threshold is expected to be between 0.0 and 1.0 (default 0.0)"
            )
            return 0.0

    def transform_graph(self, data):
        # Make graph undirected - PyTorchGeometric performs the transformation by adding reverse edges
        # (on every heterogeneous relations)
        data = T.ToUndirected()(data)
        # Manually copy labels to reverted edges
        data["target", "compound"].y_edge_index = torch.LongTensor(
            [
                data["compound", "target"].y_edge_index[1, :].tolist(),
                data["compound", "target"].y_edge_index[0, :].tolist(),
            ]
        )
        data["target", "compound"].y = torch.Tensor(data["compound", "target"].y.tolist())
        data["target", "compound"].train_idx = torch.LongTensor(data["compound", "target"].train_idx.tolist())
        data["target", "compound"].val_idx = torch.LongTensor(data["compound", "target"].val_idx.tolist())
        data["target", "compound"].test_idx = torch.LongTensor(data["compound", "target"].test_idx.tolist())
        return data

    def randomize_edges(self, seed=123, random_pAct=True):
        # Function build to randomize edges and their label, and return a 'random graph' in order to compare learning
        # processes and biological understanding of the model.
        # Here we are drawing bioactivities values randomly, instead of shuffling the existing ones.
        torch.manual_seed(seed)

        data = copy.deepcopy(self.data)

        nb_edges = data["compound", "target"].y.shape[0]  # Keep the same number of edges

        # Either pick random pAct (following a uniform distribution) or shuffling existing pAct
        if random_pAct:
            y_perm = Uniform(low=3.0, high=11.0).sample((nb_edges, 1))
        else:
            y_perm = data["compound", "target"].y[torch.randperm(nb_edges)]

        # Randomly choose nodes in relation
        data["compound", "target"].y_edge_index = torch.LongTensor(
            [
                torch.randint(low=0, high=len(self.compounds), size=(nb_edges,)).tolist(),
                torch.randint(low=0, high=len(self.targets), size=(nb_edges,)).tolist(),
            ]
        )
        data["compound", "target"].y = y_perm

        # Manually copy labels to reverted edges
        data["target", "compound"].y_edge_index = torch.LongTensor(
            [
                data["compound", "target"].y_edge_index[1, :].tolist(),
                data["compound", "target"].y_edge_index[0, :].tolist(),
            ]
        )
        data["target", "compound"].y = torch.Tensor(data["compound", "target"].y.tolist())

        return data

    def build_process(self):
        # Build and save graph
        self.data = self.build_graph()
        torch.save(self.data, os.path.join(self.data_folder, self.processed_file_names))
        return self.data

    def len(self):
        return len(self.processed_file_names)

    def get(self):
        # Build graph, or load it if it has already been saved
        if os.path.isfile(os.path.join(self.data_folder, self.processed_file_names)):
            print(
                "PDA Dataset: Loading data from {} ...".format(
                    os.path.join(self.data_folder, self.processed_file_names)
                )
            )
            self.data = self.load_data()
        else:
            print("PDA Dataset: Building data...")
            self.data = self.build_process()
        return self.data

    def get_metadata(self, use_dtis):
        # Custom metadata getter, because convolutional layers shouldn't use all target-compound for message passing
        if use_dtis:  # Here we'll use the target-compounds in the train set
            return (
                ["target", "compound"],
                [
                    ("target", "interaction", "target"),
                    ("compound", "similarity", "compound"),
                    ("compound", "interaction", "target"),
                    ("target", "rev_interaction", "compound"),
                ],
            )
        else:
            return (
                ["target", "compound"],
                [("target", "interaction", "target"), ("compound", "similarity", "compound")],
            )

    def load(self, path):
        print("PDA Dataset: Loading previous dataset from {} ...".format(path))
        self.data = torch.load(os.path.join(self.data_folder, path))
        return self.data

    def export_data(self, data, only_test=False, uniprot2seq=False, protein_graph=None, file_name=None):
        if file_name is None:
            file_name = self.processed_file_names
        # Export data to csv file
        if only_test:
            test_idx = data["compound", "target"].test_idx
            test_compounds = [data["compound"].smiles[i] for i in data["compound", "target"].y_edge_index[0][test_idx]]
            if uniprot2seq:
                protein_graph_dict = torch.load(os.path.join(self.data_folder, protein_graph))
                test_targets = [protein_graph_dict[i]["seq"] for i in data["compound", "target"].y_edge_index[1][test_idx].tolist()]
            else:
                test_targets = [data["target"].names[i] for i in data["compound", "target"].y_edge_index[1][test_idx]]          
            test_pAct = [round(x, 2) for x in data["compound", "target"].y[test_idx].flatten().tolist()]
            df = pd.DataFrame(data={"compound": test_compounds, "target": test_targets, "pAct": test_pAct})
            df.to_csv(os.path.join(self.data_folder, file_name.replace(".pt", "_test.csv")), index=False)
        else:
            all_compounds = [data["compound"].smiles[i] for i in data["compound", "target"].y_edge_index[0]]
            if uniprot2seq:
                protein_graph_dict = torch.load(os.path.join(self.data_folder, protein_graph))
                all_targets = [protein_graph_dict[i]["seq"] for i in data["compound", "target"].y_edge_index[1].tolist()]
            else:
                all_targets = [data["target"].names[i] for i in data["compound", "target"].y_edge_index[1]]
            all_pAct = [round(x, 2) for x in data["compound", "target"].y.flatten().tolist()]
            all_sets = [
                (
                    "train"
                    if i in data["compound", "target"].train_idx
                    else "val" if i in data["compound", "target"].val_idx else "test"
                )
                for i in range(len(all_pAct))
            ]
            df = pd.DataFrame(
                data={"compound": all_compounds, "target": all_targets, "Label": all_pAct, "set": all_sets}
            )
            df.to_csv(os.path.join(self.data_folder, file_name.replace(".pt", ".csv")), index=False)