import pandas as pd
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.bipartite.projection import generic_weighted_projected_graph, weighted_projected_graph
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import community as community_louvain 
import matplotlib.patches as mpatches

IMAGES_DIR = Path("../results/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def create_bipartite_graph_from_csv(csv_path):
    """
    Build a bipartite Graph where:
      - Left-side nodes are senators (bipartite=0).
      - Right-side nodes are stock tickers (bipartite=1).
      - Each edge carries two attributes:
          * 'net_amount'  → (purchase_amount - sale_amount) aggregated per (senator,ticker).
          * 'transaction' → 'purchase' or 'sale' is used only to decide color at draw‐time.
    """
    df = pd.read_csv(csv_path)
    df = df[df['ticker'].notnull()]
    df['type'] = df['type'].str.lower()
    df['type'] = df['type'].apply(lambda x: 'sale' if isinstance(x, str) and x.startswith('sale') else x)
    df = df[df['type'].isin(['purchase', 'sale'])]

    G = nx.Graph()
    senators = set(df['member'])
    tickers  = set(df['ticker'])

    # Add nodes with bipartite attribute
    G.add_nodes_from(senators, bipartite=0)
    G.add_nodes_from(tickers,  bipartite=1)

    # For each row, accumulate amount into edge attribute "net_amount"
    # If type == 'purchase', we add; if 'sale', we subtract.
    for _, row in df.iterrows():
        s = row['member']
        t = row['ticker']
        amt = float(row['amount'])  # ensure it's numeric
        sign = 1 if row['type'] == 'purchase' else -1

        if G.has_edge(s, t):
            # Increment the existing net_amount
            G[s][t]['net_amount'] += sign * amt
        else:
            G.add_edge(s, t,
                       net_amount = sign * amt,
                       transaction = row['type'])
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
    Visualize a weighted bipartite graph:
      - Node positions: senators on x=0, tickers on x=1.
      - Edge color:
          * Green if net_amount > 0 (net purchase)
          * Red   if net_amount < 0 (net sale)
      - Edge width  ∝ sqrt(abs(net_amount)) (for better visual scaling).
    """
    pos = {}
    senators = sorted(senator_nodes)
    tickers  = sorted(ticker_nodes)

    # Assign y‐positions evenly
    for i, node in enumerate(senators):
        pos[node] = (0, i)
    for i, node in enumerate(tickers):
        pos[node] = (1, i)

    plt.figure(figsize=(14, 10))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=senators,
                           node_color="skyblue",
                           label="Members",
                           node_size=1000,
                           alpha=0.8)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=tickers,
                           node_color="lightgreen",
                           label="Stocks",
                           node_size=1000,
                           alpha=0.8)

    # Build separate edge lists for purchases vs. sales (based on net_amount)
    purchase_edges = []
    sale_edges     = []
    purchase_widths = []
    sale_widths     = []

    for u, v, attr in G.edges(data=True):
        amt = attr.get('net_amount', 0)

        # Choose color by sign, width by sqrt(|amt|)
        width = (abs(amt) ** 0.5) / 250.0  # scale down so widths aren't enormous
        if amt > 0:
            purchase_edges.append((u, v))
            purchase_widths.append(width)
        else:
            sale_edges.append((u, v))
            sale_widths.append(width)

    # Draw edges with varying widths
    nx.draw_networkx_edges(G, pos,
                           edgelist=purchase_edges,
                           edge_color="green",
                           width=purchase_widths,
                           alpha=0.6)
    nx.draw_networkx_edges(G, pos,
                           edgelist=sale_edges,
                           edge_color="red",
                           width=sale_widths,
                           alpha=0.6)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Net Purchase'),
        Line2D([0], [0], color='red',   lw=2, label='Net Sale')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    filename = title.lower().replace(" ", "_") + ".png"
    filepath = IMAGES_DIR / filename
    plt.savefig(filepath, dpi=300)
    plt.close()
    

def visualize_projection_graph(P, membership, title):
    """
    Draw the member-member projection P with nodes colored by community.
    Adds a legend mapping community ID → color.
    """
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(P, seed=42)

    # Determine how many distinct communities we have
    communities = sorted(set(membership.values()))
    num_comms = len(communities)

    # Generate a color for each community from the tab20 colormap
    cmap = plt.cm.tab20
    community_to_color = {
        comm: cmap(comm % 20)  # wrap around if >20
        for comm in communities
    }

    # Draw nodes, one community at a time (so we can collect handles for the legend)
    legend_handles = []
    for comm in communities:
        nodes_in_comm = [n for n, c in membership.items() if c == comm]
        nx.draw_networkx_nodes(
            P,
            pos,
            nodelist=nodes_in_comm,
            node_color=[community_to_color[comm]],
            node_size=1200,
            alpha=0.9,
            label=f"Community {comm}"
        )
        legend_handles.append(
            mpatches.Patch(color=community_to_color[comm], label=f"Community {comm}")
        )

    # Draw all edges in light gray
    nx.draw_networkx_edges(P, pos, edge_color="lightgray", alpha=0.3)
    nx.draw_networkx_labels(P, pos, font_size=12)

    # Add the legend outside the plot
    plt.legend(handles=legend_handles, loc="upper right", title="Communities")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    filename = title.lower().replace(" ", "_") + ".png"
    filepath = IMAGES_DIR / filename
    plt.savefig(filepath, dpi=300)
    plt.close()


def visualize_community_meta_graph(P, communities, membership, title):
    """
    Build and draw the meta‐graph where each node is a community. 
    Node size ∝ community size, node color matches the color used in visualize_projection_graph.
    Adds a legend mapping community ID → color.
    """
    # First, create the meta‐graph G_meta
    G_meta = nx.Graph()
    community_map = membership

    # Add one node per community and store its size
    for idx, comm in enumerate(communities):
        G_meta.add_node(idx, size=len(comm))

    # Count inter‐community edges
    for u, v in P.edges():
        cu = community_map[u]
        cv = community_map[v]
        if cu == cv:
            continue
        if G_meta.has_edge(cu, cv):
            G_meta[cu][cv]["weight"] += 1
        else:
            G_meta.add_edge(cu, cv, weight=1)

    # Layout for the meta‐graph
    pos = nx.spring_layout(G_meta, seed=42)

    # Reuse the same colormap mapping as in visualize_projection_graph
    cmap = plt.cm.tab20
    community_ids = sorted(community_map.values())
    distinct_comms = sorted(set(community_ids))
    community_to_color = {
        comm: cmap(comm % 20)
        for comm in distinct_comms
    }

    # Prepare legend handles
    legend_handles = [
        mpatches.Patch(color=community_to_color[comm], label=f"Community {comm}")
        for comm in distinct_comms
    ]

    # Draw each community node individually, so that color matches
    node_sizes = []
    node_colors = []
    for comm_idx in sorted(G_meta.nodes()):
        size = G_meta.nodes[comm_idx]["size"] * 100
        node_sizes.append(size)
        node_colors.append(community_to_color[comm_idx])

    # Edge widths scaled by weight
    edge_widths = [G_meta[u][v]["weight"] / 100 for u, v in G_meta.edges()]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(
        G_meta,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8
    )
    nx.draw_networkx_edges(
        G_meta,
        pos,
        width=edge_widths,
        alpha=0.4
    )
    nx.draw_networkx_labels(G_meta, pos, font_size=10)

    # Legend (one handle per community)
    plt.legend(handles=legend_handles, loc="upper right", title="Communities")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

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
    filename = title.lower().replace(" ", "_") + ".png"
    filepath = IMAGES_DIR / filename
    plt.savefig(filepath, dpi=300)
    plt.close()

    
def compute_modularity(P, node_to_comm):
    """
    Compute the Louvain modularity of the given partition (node_to_comm) on graph P.
    """
    return community_louvain.modularity(node_to_comm, P, weight="weight")


def main():
    # Plot trades.csv graph
    csv_path = Path("../data/cleaned/2014-2023/stocks.csv")
    df = pd.read_csv(csv_path)
    
    H, senators, tickers = create_bipartite_graph_from_csv(csv_path)
    print(f"Original graph: {len(senators)} members, {len(tickers)} tickers, {H.number_of_edges()} edges")
    
    G, filtered_senators, filtered_tickers = filter_by_degree(H, df, senators, tickers, min_degree=10, min_tx=300)
    print(f"Filtered graph: {len(filtered_senators)} members, {len(filtered_tickers)} tickers, {G.number_of_edges()} edges")

    visualize_bipartite_graph(
        G, filtered_senators, filtered_tickers,
        title="Filtered Graph-Members and Tickers"
    )

    P, communities, membership = get_communities(H, filtered_senators)

    for i, comm in enumerate(communities):
        print(f"Community {i} (size={len(comm)}): {comm}")
        
    # Compute and print the Louvain modularity score
    modularity_score = compute_modularity(P, membership)
    print(f"\nLouvain modularity (member-member projection): {modularity_score:.4f}")

    visualize_projection_graph(P, membership, "Member-Member Projection")
    visualize_community_meta_graph(P, communities, membership, "Community-Level Trading Graph")

    # Centrality metrics
    deg_cent = nx.degree_centrality(P)
    btw_cent = nx.betweenness_centrality(P, normalized=True)
    eig_cent = nx.eigenvector_centrality_numpy(P, weight='weight')
    cls_cent = nx.closeness_centrality(P)

    # Combine into DataFrame
    df_centrality = pd.DataFrame({
        'name': list(P.nodes()),
        'degree': [deg_cent[n] for n in P.nodes()],
        'betweenness': [btw_cent[n] for n in P.nodes()],
        'eigenvector': [eig_cent[n] for n in P.nodes()],
        'closeness': [cls_cent[n] for n in P.nodes()],
    })

    # Sort and show top results
    print("\nSenators by centrality (sorted by closeness):")
    print(
        df_centrality
        .sort_values('closeness', ascending=False)
        .to_string(index=False)
    )
    
    P_ticker = weighted_projected_graph(G, filtered_tickers)

    # Compute four centralities on P_ticker
    deg_cent_tic = nx.degree_centrality(P_ticker)
    btw_cent_tic = nx.betweenness_centrality(P_ticker, normalized=True)
    eig_cent_tic = nx.eigenvector_centrality_numpy(P_ticker, weight='weight')
    cls_cent_tic = nx.closeness_centrality(P_ticker)

    df_centrality_tic = pd.DataFrame({
        'ticker':     list(P_ticker.nodes()),
        'degree':     [deg_cent_tic[n] for n in P_ticker.nodes()],
        'betweenness':[btw_cent_tic[n] for n in P_ticker.nodes()],
        'eigenvector':[eig_cent_tic[n] for n in P_ticker.nodes()],
        'closeness':  [cls_cent_tic[n] for n in P_ticker.nodes()],
    })

    print("\nStocks by centrality (sorted by eigenvector):")
    print(
        df_centrality_tic
        .sort_values('eigenvector', ascending=False)
        .to_string(index=False)
    )

    plot_industry_by_community(csv_path, communities, 
                               title="Industry Purchases by Community")

if __name__ == '__main__':
    main()
