import pandas as pd
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.bipartite.projection import generic_weighted_projected_graph, weighted_projected_graph
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

IMAGES_DIR = Path("../results/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def create_bipartite_graph_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['ticker'].notnull()]
    df['type'] = df['type'].str.lower()
    df['type'] = df['type'].replace({'sale_partial': 'sale', 'sale_full': 'sale'})
    df = df[df['type'].isin(['purchase', 'sale'])]

    G = nx.Graph()
    senators = set(df['member'])
    tickers = set(df['ticker'])

    G.add_nodes_from(senators, bipartite=0)
    G.add_nodes_from(tickers, bipartite=1)

    for _, row in df.iterrows():
        G.add_edge(row['member'], row['ticker'], transaction=row['type'])

    return G, senators, tickers



def filter_by_degree(G, df, senator_nodes, ticker_nodes, min_degree, min_tx):
    member_counts = df['member'].value_counts()
    active_members = set(member_counts[member_counts >= min_tx].index)

    senator_nodes_filtered = {s for s in senator_nodes if s in active_members}

    G_pruned = G.subgraph(senator_nodes_filtered.union(ticker_nodes)).copy()

    ticker_nodes_pruned = {t for t in ticker_nodes if t in G_pruned.nodes()}
    senator_nodes_pruned = senator_nodes_filtered  

    ticker_deg = {t: G_pruned.degree(t) for t in ticker_nodes_pruned}
    filtered_tickers = {t for t, deg in ticker_deg.items() if deg >= min_degree}

    filtered_senators = {
        nbr
        for t in filtered_tickers
        for nbr in G_pruned.neighbors(t)
        if nbr in senator_nodes_pruned
    }

    sub_nodes = filtered_senators.union(filtered_tickers)
    H = G_pruned.subgraph(sub_nodes).copy()

    return H, filtered_senators, filtered_tickers



def get_communities(G, senator_nodes):
    P = nx.projected_graph(G, senator_nodes)

    # Use unweighted Louvain
    communities = community.louvain_communities(P, weight=None, seed=42, resolution=1.0)
    membership = {node: idx for idx, comm in enumerate(communities) for node in comm}
    return P, communities, membership

def visualize_bipartite_graph(G, senator_nodes, ticker_nodes, title):
    """
    Visualize a bipartite graph with green edges for purchases and red edges for sales.
    """
    pos = {}
    senators = sorted(senator_nodes)
    tickers = sorted(ticker_nodes)

    for i, node in enumerate(senators):
        pos[node] = (0, i)
    for i, node in enumerate(tickers):
        pos[node] = (1, i)

    plt.figure(figsize=(14, max(len(senators), len(tickers)) // 3))

    nx.draw_networkx_nodes(G, pos, nodelist=senators, node_color="skyblue", label="Members", node_size=200)
    nx.draw_networkx_nodes(G, pos, nodelist=tickers, node_color="lightgreen", label="Stocks", node_size=200)

    # Separate edges by type
    purchase_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("transaction") == "purchase"]
    sale_edges     = [(u, v) for u, v, d in G.edges(data=True) if d.get("transaction") == "sale"]

    nx.draw_networkx_edges(G, pos, edgelist=purchase_edges, edge_color="green", alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=sale_edges, edge_color="red", alpha=0.6)

    nx.draw_networkx_labels(G, pos, font_size=7)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Purchase'),
        Line2D([0], [0], color='red', lw=2, label='Sale'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    filename = title.lower().replace(" ", "_") + ".png"
    filepath = IMAGES_DIR / filename
    plt.savefig(filepath, dpi=300)
    plt.close()
    
def visualize_community_meta_graph(P, communities, membership, title):
    # Build meta-graph
    G_meta = nx.Graph()

    # Map each node to its community
    community_map = membership

    # Create nodes for each community
    for idx, comm in enumerate(communities):
        G_meta.add_node(idx, size=len(comm))

    # Count edges between communities
    for u, v in P.edges():
        cu = community_map[u]
        cv = community_map[v]
        if cu == cv:
            continue  # skip intra-community
        if G_meta.has_edge(cu, cv):
            G_meta[cu][cv]['weight'] += 1
        else:
            G_meta.add_edge(cu, cv, weight=1)

    # Layout and draw
    pos = nx.spring_layout(G_meta, seed=42)
    sizes = [G_meta.nodes[n]['size'] * 100 for n in G_meta.nodes()]
    edge_widths = [G_meta[u][v]['weight'] / 100 for u, v in G_meta.edges()]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G_meta, pos, node_size=sizes, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G_meta, pos, width=edge_widths, alpha=0.4)
    nx.draw_networkx_labels(G_meta, pos, font_size=10)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    filename = title.lower().replace(" ", "_") + ".png"
    filepath = IMAGES_DIR / filename
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_industry_by_community(csv_path, communities, title):
    # 1) Load raw data
    df = pd.read_csv(csv_path)
    df = df[df['ticker'].notnull()]
    df['type'] = df['type'].str.lower()\
                         .replace({'sale_partial':'sale','sale_full':'sale'})
    # only purchases
    df = df[df['type'] == 'purchase']
    
    # 2) Build a Series of (community → industry counts)
    comm_counts = {}
    for comm_id, members in enumerate(communities):
        # select rows where member is in this community
        sel = df[df['member'].isin(members)]
        comm_counts[comm_id] = sel['industry'].value_counts()
    
    # 3) Turn into DataFrame (rows=communities, cols=industries)
    comm_df = pd.DataFrame(comm_counts).T.fillna(0).astype(int)
    
    # 4) Keep only overall top-10 industries
    total = comm_df.sum(axis=0)
    top10 = total.nlargest(10).index
    comm_df = comm_df[top10]
    
    # 5) Plot stacked bar
    ax = comm_df.plot(
        kind='bar',
        stacked=True,
        figsize=(10, 6)
    )
    ax.set_xlabel("Community")
    ax.set_ylabel("Number of Purchases")
    ax.set_title("Industry Purchases by Louvain Community")
    ax.legend(title="Industry", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    filename = title.lower().replace(" ", "_") + ".png"
    filepath = IMAGES_DIR / filename
    plt.savefig(filepath, dpi=300)
    plt.close()


def visualize_projection_graph(P, membership, title):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(P, seed=42)
    colors = [membership[node] for node in P.nodes()]
    nx.draw_networkx_nodes(P, pos, node_color=colors, cmap=plt.cm.tab20, node_size=200, alpha=0.9)
    nx.draw_networkx_edges(P, pos, alpha=0.3)
    nx.draw_networkx_labels(P, pos, font_size=7)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    filename = title.lower().replace(" ", "_") + ".png"
    filepath = IMAGES_DIR / filename
    plt.savefig(filepath, dpi=300)
    plt.close()


def main():
    # Plot trades.csv graph
    csv_path = Path("../data/cleaned/2014-2023/stocks.csv")
    df = pd.read_csv(csv_path)
    
    H, senators, tickers = create_bipartite_graph_from_csv(csv_path)
    print(f"Original graph: {len(senators)} members, {len(tickers)} tickers, {H.number_of_edges()} edges")
    
    visualize_bipartite_graph(
        H, senators, tickers,
        title="Bipartite Graph-Members and Tickers"
    )

    G, filtered_senators, filtered_tickers = filter_by_degree(H, df, senators, tickers, min_degree=15, min_tx=100)
    print(f"Filtered graph: {len(filtered_senators)} members, {len(filtered_tickers)} tickers, {G.number_of_edges()} edges")

    visualize_bipartite_graph(
        G, filtered_senators, filtered_tickers,
        title="Filtered Graph-Members and Tickers"
    )

    P, communities, membership = get_communities(H, filtered_senators)

    for i, comm in enumerate(communities):
        print(f"Community {i} (size={len(comm)}): {comm}")

    visualize_projection_graph(P, membership, "Member-Member Projection")
    visualize_community_meta_graph(P, communities, membership, "Community-Level Trading Graph")

    # Centrality metrics
    deg_cent = nx.degree_centrality(P)
    btw_cent = nx.betweenness_centrality(P, normalized=True)
    eig_cent = nx.eigenvector_centrality_numpy(P, weight='weight')
    cls_cent = nx.closeness_centrality(P)

    # Combine into DataFrame
    df_centrality = pd.DataFrame({
        'ticker': list(P.nodes()),
        'degree': [deg_cent[n] for n in P.nodes()],
        'betweenness': [btw_cent[n] for n in P.nodes()],
        'eigenvector': [eig_cent[n] for n in P.nodes()],
        'closeness': [cls_cent[n] for n in P.nodes()],
    })

    # Sort and show top results
    print("\nSenators by centrality (sorted by betweenness):")
    print(
        df_centrality
        .sort_values('betweenness', ascending=False)
        .to_string(index=False)
    )

    plot_industry_by_community(csv_path, communities, 
                               title="Industry Purchases by Community")


if __name__ == '__main__':
    main()
