import pandas as pd
import numpy as np
import argparse
from graphviz import Digraph

class Node:

    def __init__(self, name: str) -> None:
        self.name = name
        self.incoming_edges = set()
        self.outgoing_edges = set()


    def add_incoming_edge(self, edge):
        self.incoming_edges.add(edge)


    def add_outgoing_edge(self, edge):
        self.outgoing_edges.add(edge)


class Edge:

    def __init__(self, start, dest, val) -> None:
        self.start = start
        self.dest = dest
        self.val = val

class AdjacencyGraph:

    def __init__(self):
        self.names = set()
        self.nodes = set()
        self.edges = set()

    
    def add_node(self, node):
        if node.name not in self.names:
            self.nodes.add(node)
            self.names.add(node.name)
            return True
        return False

    
    def add_edge(self, edge):
        if (not any([e.dest == edge.dest for e in edge.start.outgoing_edges])) or \
           (not any([e.start == edge.start for e in edge.dest.incoming_edges])):
            self.edges.add(edge)
            edge.start.add_outgoing_edge(edge)
            edge.dest.add_incoming_edge(edge)
            return True
        return False
    

    def get_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')
    # parser.add_argument('-m', '--male', type=bool, default=False,
                        # action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.datapath)
    df.columns = [str(c).strip() for c in df.columns]
    adj_graph = AdjacencyGraph()
    adj_graph.add_node(Node('sex_Male'))
    adj_graph.add_node(Node('sex_Female'))
    frontier = ['sex_Male', 'sex_Female']
    while frontier:
        column = frontier.pop(0)
        if not column.startswith('salary'):
            indices = np.flatnonzero(df[column])
            print(indices)
            for i in indices:
                print(i)
                i_column = str(df.columns[i])
                print(i_column)
                if adj_graph.add_node(Node(i_column)):
                    frontier.append(i_column)
                adj_graph.add_edge(Edge(adj_graph.get_node(column), adj_graph.get_node(i_column), df[column][i]))
    render_graph = Digraph(args.datapath)
    names_to_labels = dict()
    for i, node in enumerate(adj_graph.nodes):
        label = chr(i + ord('A')) if i < 26 else chr(i + ord('a') - 26)
        render_graph.node(label, node.name)
        names_to_labels[node.name] = label
    for edge in adj_graph.edges:
        start_label = names_to_labels[edge.start.name]
        dest_label = names_to_labels[edge.dest.name]
        render_graph.edge(start_label, dest_label, label=str(edge.val))
    print(render_graph.source)
    render_graph.render(directory='.').replace('\\', '/')


if __name__ == "__main__":
    main()