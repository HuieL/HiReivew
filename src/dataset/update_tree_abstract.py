import os
import torch
from tqdm import tqdm
import time
from scholarly import scholarly, ProxyGenerator


def setup_proxy():
    pg = ProxyGenerator()
  
    success = pg.FreeProxies()
    if not success:
        print("Failed to set up proxy. Requests may get blocked.")
    scholarly.use_proxy(pg)

def get_scholar_query(query):
    try:
        search_query = scholarly.search_pubs(query)
        first_result = next(search_query)
        abstract = first_result['bib'].get('abstract', 'No abstract available')
        return abstract
    except StopIteration:
        return None
    except Exception as e:
        print(f"Error retrieving abstract: {e}")
        return None


def update_graph_with_abstract(graph):
    if len(graph.arxiv_ids) != len(graph.titles): 
        return None
    if not hasattr(graph, 'abstract'):
        graph.abstract = [None] * len(graph.titles)
    
    updated_count = 0
    for i in tqdm(range(len(graph.titles)), desc="Updating abstracts", leave=False):
        arxiv_id, title, abstract = graph.arxiv_ids[i], graph.titles[i], graph.abstract[i]
        if abstract is None:
            if arxiv_id:
                new_abstract = get_arxiv_abstract(arxiv_id)
            else:
                new_abstract = get_scholar_query(title)
              
                graph.abstract[i] = new_abstract
                updated_count += 1
                time.sleep(1)
    
    print(f"Updated {len([ab for ab in graph.abstract if ab is not None])} abstracts out of {len(graph.arxiv_ids)} total entries.")
    return graph

def update_graphs_in_directory(directory):
    updated_graphs = 0
    total_graphs = 0
    
    for filename in os.listdir(directory):
        total_graphs += 1
        file_path = os.path.join(directory, filename)
      
        graph = torch.load(file_path)
        if graph is not None:
            updated_graph = update_graph_with_abstract(graph)
            torch.save(updated_graph, file_path)
            
            updated_graphs += 1
    
    print(f"Updated {updated_graphs} out of {total_graphs} graphs in {directory}")

if __name__ == "__main__":
  directory = "./dataset/trees"
  update_graphs_in_directory(directory)
