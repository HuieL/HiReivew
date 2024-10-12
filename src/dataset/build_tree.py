from tqdm import tqdm
import requests
import re
import tarfile
import os
import glob
import xml.etree.ElementTree as ET
import urllib.parse
import tempfile
import time
import torch
from torch_geometric.data import Data
from src.utils.extract_arxiv import get_arxiv_id, download_arxiv_source, find_main_tex_file, find_reference_files, parse_bib_content, parse_bbl_content, get_arxiv_main_text, get_arxiv_abstract


def get_arxiv_ids(titles):
    arxiv_ids = []
    not_found = 0
    print("Processing arixv ids ...")
    for title in tqdm(titles):
        if not_found > 25: break
        try:
            arxiv_id = get_arxiv_id(title)
            arxiv_ids.append(arxiv_id)
            time.sleep(1) 
            not_found = 0
        except Exception:
            arxiv_ids.append(None)
            not_found+=1
    return arxiv_ids

def extract_reference_titles(arxiv_id):
    cited_titles = []
    bib_content = None
    try:
        tar_file = download_arxiv_source(arxiv_id)
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.extractall(path=temp_dir)
            except tarfile.ReadError:
                with tarfile.open(tar_file, 'r:') as tar:
                    tar.extractall(path=temp_dir)
            
            main_tex_file = find_main_tex_file(temp_dir)
            
            if not main_tex_file:
                print(f"Could not find main .tex file for {arxiv_id}")
                return cited_titles, bib_content

            with open(main_tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                tex_content = process_input_commands(f.read(), os.path.dirname(main_tex_file))
            
            ref_files = find_reference_files(tex_content)
            if not ref_files:
                print(f"No reference files found in the processed .tex content for {arxiv_id}")
                return cited_titles, bib_content

            for ref_file in ref_files:
                ref_path = os.path.join(os.path.dirname(main_tex_file), ref_file)
                if not os.path.exists(ref_path):
                    for ext in ['.bib', '.bbl']:
                        test_path = ref_path + ext
                        if os.path.exists(test_path):
                            ref_path = test_path
                            break
                
                if os.path.exists(ref_path):
                    try:
                        with open(ref_path, 'r', encoding='utf-8', errors='ignore') as ref_file:
                            content = ref_file.read()
                            if ref_path.endswith('.bib'):
                                titles = parse_bib_content(content)
                                bib_content = content  # Store the entire bib content
                            elif ref_path.endswith('.bbl'):
                                titles = parse_bbl_content(content)
                                bib_content = content
                            else:
                                print(f"Unrecognized reference file type: {ref_path}")
                                continue
                            cited_titles.extend(titles)
                    except UnicodeDecodeError:
                        print(f"Could not read file {ref_path} as UTF-8")

    except Exception as e:
        print(f"An error occurred while processing {arxiv_id}: {str(e)}")
    finally:
        if 'tar_file' in locals() and os.path.exists(tar_file):
            os.unlink(tar_file)

    return cited_titles, bib_content

def construct_citation_graph(root_arxiv_id, root_title, cited_titles, cited_arxiv_ids, root_bib_found):

    # Create a list of all valid arXiv IDs (including root) and their corresponding titles
    all_arxiv_ids = [root_arxiv_id] + list(cited_arxiv_ids)
    all_titles = [root_title] + list(cited_titles)

    id_to_index = {id: i for i, id in enumerate(all_arxiv_ids)}
    edge_index = [[0, id_to_index[id]] for id in cited_arxiv_ids]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(edge_index=edge_index)
    data.arxiv_ids = all_arxiv_ids
    data.titles = all_titles
    data.bib_found = root_bib_found

    return data

def build_citation_tree_with_ids(root_arxiv_id, paper_title):
    try:
        main_text = get_arxiv_main_text(root_arxiv_id)
        abstract = get_arxiv_abstract(root_arxiv_id)

        cited_titles, root_bib_found = extract_reference_titles(root_arxiv_id)
        if len(cited_titles) > 1000:
            return None
        cited_arxiv_ids = get_arxiv_ids(cited_titles)
        print(f"Found citation on arxivs: {len([(i, id) for i, id in enumerate(cited_arxiv_ids) if id is not None])}/{len(cited_arxiv_ids)}")

        graph = construct_citation_graph(root_arxiv_id, paper_title, cited_titles, cited_arxiv_ids, root_bib_found)

        graph.content = [main_text]
        graph.abstract = [abstract]
        graph.title = [paper_title]

        for _, title in zip(cited_arxiv_ids, cited_titles):
                graph.content.append(None)
                graph.abstract.append(None)
                graph.title.append(title)

        return graph
    except Exception:
        return None

def process_papers_from_file(file_path = "./survey_info.txt", output_folder = "./dataset/trees"):
    os.makedirs(output_folder, exist_ok=True)
    papers = []
    with open(file_path, 'r') as f:
        for line in f:
            arxiv_id, title = line.strip().split(', ', 1)
            papers.append((arxiv_id, title))

    for arxiv_id, title in papers:
        output_file = os.path.join(output_folder, f"{arxiv_id}.pt")
        if os.path.exists(output_file):
            print(f"Graph for {arxiv_id} already exists. Skipping.")
            continue

        graph = build_citation_tree_with_ids(arxiv_id, title)
        
        if graph is not None and len(graph.title)>1:
            torch.save(graph, output_file)
            print(f"Saved graph for {arxiv_id}")
        else:
            print(f"Failed to build graph for {arxiv_id}")

def clean_pyg_graphs(folder_path = "./dataset/trees"):
    pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    deleted_count = 0
    
    for file in tqdm(pt_files, desc="Processing graphs"):
        file_path = os.path.join(folder_path, file)
        try:
            graph = torch.load(file_path)

            if isinstance(graph, Data) and hasattr(graph, 'title') and len(graph.title) == 1:
                os.remove(file_path)
                deleted_count += 1
                print(f"Deleted {file} - len(graph.title) == 1")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    

if __name__ == "__main__":
    process_papers_from_file()
    clean_pyg_graphs()
