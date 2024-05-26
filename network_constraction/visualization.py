import networkx as nx
import logging
import matplotlib.pyplot as plt

# 设置日志记录配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='net.log')

def load_and_visualize_graph(graphml_path):
    # Load GraphML file
    G = nx.read_graphml(graphml_path)
    logging.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Simplify graph: remove self-loops and multiple edges
    G = nx.Graph(G)  # Convert to undirected graph, automatically merges multiple edges and removes self-loops
    logging.info(f"Simplified graph to {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Calculate and print average edge weight
    total_weight = sum(weight for u, v, weight in G.edges(data='weight', default=1))
    average_weight = total_weight / G.number_of_edges() if G.number_of_edges() > 0 else 0
    logging.info(f"Average edge weight: {average_weight:.2f}")

    # Calculating network properties
    density = nx.density(G)
    logging.info(f"Network density: {density:.4f}")

    degree_centrality = nx.degree_centrality(G)
    avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)
    logging.info(f"Average degree centrality: {avg_degree_centrality:.4f}")

    closeness_centrality = nx.closeness_centrality(G)
    avg_closeness_centrality = sum(closeness_centrality.values()) / len(closeness_centrality)
    logging.info(f"Average closeness centrality: {avg_closeness_centrality:.4f}")

    betweenness_centrality = nx.betweenness_centrality(G)
    avg_betweenness_centrality = sum(betweenness_centrality.values()) / len(betweenness_centrality)
    logging.info(f"Average betweenness centrality: {avg_betweenness_centrality:.4f}")

# Specify your GraphML file path
graphml_path = "/scratch/iu60/xs5813/graph_Aus_mask_3year_thresh2_same_t_and_r.graphml"
load_and_visualize_graph(graphml_path)
