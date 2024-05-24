import pandas as pd
import numpy as np
import argparse
from graphviz import Digraph
from create_adjacency_matrix import AdjacencyGraph, Node, Edge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')
    # parser.add_argument('-m', '--male', default=False,
    #                     action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.datapath)
    df.columns = [str(c).strip() for c in df.columns]
    adj_graph = AdjacencyGraph()
    start_features = ['sex_Male', 'sex_Female']
    for start_feature in start_features:
        node = Node(start_feature)
        adj_graph.add_node(node)
        start_idx = list(df.columns).index(start_feature)
        for i in range(df.shape[1]):
            incoming_val = df.iloc[start_idx, i]
            outgoing_val = df.iloc[i, start_idx]
            new_node = Node(df.columns[i])
            if abs(incoming_val) > 0.05 and abs(incoming_val) < 1000000:
                adj_graph.add_node(new_node)
                adj_graph.add_edge(Edge(adj_graph.get_node(df.columns[i]), adj_graph.get_node(node.name), incoming_val))
            elif abs(outgoing_val) > 0.05 and abs(outgoing_val) < 1000000:
                adj_graph.add_node(new_node)
                adj_graph.add_edge(Edge(adj_graph.get_node(node.name), adj_graph.get_node(df.columns[i]), outgoing_val))
    render_graph = Digraph(args.datapath)
    names_to_labels = dict()
    for i, node in enumerate(adj_graph.nodes):
        style = 'filled' if node.name.startswith('sex') else ''
        label = chr(i + ord('A')) if i < 26 else chr(i + ord('a') - 26)
        render_graph.node(label, node.name, fontsize="20pt", style=style, fillcolor='red')
        names_to_labels[node.name] = label
    for edge in adj_graph.edges:
        start_label = names_to_labels[edge.start.name]
        dest_label = names_to_labels[edge.dest.name]
        render_graph.edge(start_label, dest_label, label=str(edge.val), fontsize="20")
    print(render_graph.source)
    render_graph = render_graph.unflatten(stagger=14)
    # if args.datapath == 'causal_graph.csv':
    #     render_graph = render_graph.unflatten(stagger=14)
    # else:
    #     render_graph = render_graph.unflatten(stagger=2)
    render_graph.render(directory='.').replace('\\', '/')


if __name__ == "__main__":
    main()