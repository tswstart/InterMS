import json
import logging
import os

import pandas as pd
import requests
from tqdm import tqdm

from src.utils import dump_to_json


def exclude_nonhuman_proteins(proteins, human_path=None, json_path=None):
    # Check which protein is human, and exclude the ones that are not
    if (
        json_path
        and os.path.isfile(json_path)
        and "human_proteins" in json.loads(open(json_path, encoding="utf-8").read())
    ):
        return json.loads(open(json_path, encoding="utf-8").read())["human_proteins"]
    else:
        human_proteins = []
        if human_path is not None:
            human_df = pd.read_csv(human_path)
            for protein in tqdm(proteins, desc='Exclude Nonhuman Proteins from human_df:'):
                if protein in human_df['id'].values:
                    human_proteins.append(protein)
        else:
            for protein in tqdm(proteins, desc='Exclude Nonhuman Proteins from uniprot_api:'):
                try:
                    url = "https://rest.uniprot.org/uniprotkb/search?query=accession:" + "{}&format=tsv&fields=organism_id".format(protein)
                    df = pd.read_csv(url, delimiter="\t")
                    if len(df) == 0:
                        logging.warning(
                            "UniProt: UniProtID {} has been redirected - please manually update your database!".format(
                                protein
                            )
                        )
                    elif len(df) > 1:
                        raise ValueError("UniProt: UniProtID {} returns ambiguous results.".format(protein))
                    else:
                        if df["Organism (ID)"].iloc[0] == 9606:  # Protein is human
                            human_proteins.append(protein)
                except Exception:
                    logging.warning("UniProt: UniProtID {} is invalid!".format(protein))
                    continue

        if json_path:
            dump_to_json({"human_proteins": human_proteins}, json_path)
        return human_proteins


def download_uniprot_fasta(proteins, data_folder):
    """
    Download FASTA sequences of the given proteins (human: organism_id=9606) from UniProt.
    Skip the download if a .fasta file with the same name already exists locally.
    Progress is shown via a tqdm progress bar.
    """
    print("\nDownloading FASTA files from UniProt...")

    # Make sure the folder for saving fasta files exists
    fasta_dir = os.path.join(data_folder, "fasta")
    if not os.path.isdir(fasta_dir):
        os.makedirs(fasta_dir)

    # Show progress with tqdm
    for uniprot_id in tqdm(proteins, desc="FASTA Download"):
        fasta_path = os.path.join(fasta_dir, uniprot_id + ".fasta")

        # Skip the download if the file already exists
        if os.path.isfile(fasta_path):
            continue

        # Build the download URL
        url = (
            "https://rest.uniprot.org/uniprotkb/search?query="
            + f"{uniprot_id}+AND+organism_id:9606&format=fasta&compressed=false"
        )

        # Download the FASTA content
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(fasta_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
        else:
            # If the request fails, print or log the failure info
            print(f"WARNING: Failed to download {uniprot_id}, status code={resp.status_code}")

    print("Download completed!")


def retrieve_protein_classes(proteins):
    # Retrieve Protein classes from UniProtID using UniProt keywords
    # This file has been downloaded directly from the UniProt website
    uniprot_keyword_df = pd.read_csv("uniprot-keywords-table.tsv", delimiter="\t")
    keyword_categories = [
        "Biological process",
        "Molecular function",
        "Disease",
        "Cellular component",
    ]  # We are interested in 4 different categories only
    category2keywords = {
        c: uniprot_keyword_df[uniprot_keyword_df.Category == c].Name.unique().tolist() for c in keyword_categories
    }
    protein_class_df = pd.DataFrame(columns=keyword_categories, index=proteins)
    for protein in proteins:
        url = (
            "https://rest.uniprot.org/uniprotkb/search?query=accession:"
            + "{}+AND+organism_id:9606&format=tsv&fields=keyword".format(protein)
        )
        df = pd.read_csv(url, delimiter="\t")
        keywords_protein_list = df.Keywords.loc[0].split(";")
        protein_class_df.loc[protein] = {
            c: [k for k in keywords_protein_list if k in category2keywords[c]] for c in keyword_categories
        }
    return protein_class_df
