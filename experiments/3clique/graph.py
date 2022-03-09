from typing import Dict, List, Set, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


class Node:
    name: int
    features: Tensor
    label: int
    in_train_mask: bool
    in_test_mask: bool
    in_val_mask: bool

    def __init__(self, name: int, features: Tensor):
        """Create a new node based on it's name (index) and `features` which is 1xF vector"""
        self.name = name
        self.features = features


class Graph:
    nodes: List[Node]
    adjacency_list: Dict[Node, Set[Node]]

    def __init__(self, edge_index: Tensor, feature_matrix: Tensor):
        """
        Create a graph from `edge_index` and `feature_matrix`.
        `edge_index[0]` is a list of source nodes and `edge_index[1]` is a list of destination nodes;
        `feature_matrix` is an NxF matrix where N is the number of nodes and F is the number of features of each node
        """
        self.nodes = []
        self.adjacency_list = {}

        nodes_num = len(edge_index[0])

        for i in range(nodes_num):
            # find nodes associated with numbers
            source = self.__find_node_or_create(int(edge_index[0][i]), feature_matrix)
            destination = self.__find_node_or_create(int(edge_index[1][i]), feature_matrix)

            # add edges to `adjacency_list`
            self.__add_edge(source, destination)
            self.__add_edge(destination, source)

    def __add_edge(self, source: Node, destination: Node):
        """Adds an edge (source, destination) to adjacency list"""
        # if key doesnt exist, create one with default value
        if self.adjacency_list.get(source, None) is None:
            self.adjacency_list[source] = set()

        self.adjacency_list[source].add(destination)

    def __find_node_or_create(self, name: int, feature_matrix: Tensor) -> Node:
        """Returns an existing node by name or creates one"""
        node = self.__find_node_by_name(name)
        if not node:
            node = Node(name, feature_matrix[name])
            self.nodes.append(node)
            self.adjacency_list[node] = set()
        return node

    def __find_node_by_name(self, name: int) -> Optional[Node]:
        """Finds a node by name. Returns `None` if node is not found"""
        for node in self.nodes:
            if node.name == name:
                return node

    def edge_exists(self, source: Node, destination: Node) -> bool:
        return destination in self.adjacency_list[source]

    def nodes_form_3clique(self, a: Node, b: Node, c: Node) -> bool:
        """Returns true if 3 given nodes form a 3-clique"""
        ab = self.edge_exists(a, b)
        ac = self.edge_exists(a, c)
        bc = self.edge_exists(b, c)
        return ab and ac and bc

    def get_all_3cliques(self) -> List[Tuple[Node, Node, Node]]:
        """
        Returns a list of tuples where nodes form a 3-clique.
        **Warning:** 3-cliques might intersect (one node can belong to multiple tuples)
        """
        result = []

        for i in range(len(self.nodes) - 2):
            for j in range(i, len(self.nodes) - 1):
                # define first 2 nodes
                a, b = self.nodes[i], self.nodes[j]
                # if there's no edge between them, don't check the third
                if not self.edge_exists(a, b):
                    continue

                for k in range(j, len(self.nodes)):
                    # define the 3rd node from possible clique
                    c = self.nodes[k]
                    if self.nodes_form_3clique(a, b, c):
                        result.append((a, b, c))

        return result

    def replace_3clique_with_node(self, clique: Tuple[Node, Node, Node]):
        """Replaces given 3 nodes with another node sharing features and edges of given nodes"""
        a, b, c = clique
        neighbors = self.adjacency_list[a] | self.adjacency_list[b] | self.adjacency_list[c]
        neighbors.remove(a)
        neighbors.remove(b)
        neighbors.remove(c)

        # remove all edges coming to these nodes
        for source in clique:
            for destination in self.adjacency_list[source]:
                self.adjacency_list[destination].remove(source)
            self.adjacency_list.pop(source)
            self.nodes.remove(source)

        # crate a "general" node
        common_features: Tensor = (a.features + b.features + c.features) / 3
        new_node_name = a.name
        common_node = Node(new_node_name, common_features)

        self.nodes.append(common_node)
        for neighbor in neighbors:
            self.__add_edge(common_node, neighbor)

    def __rename_nodes(self, old_data: Data):
        """Generates new nodes to all the nodes"""
        for ind, node in enumerate(self.nodes):
            node.label = int(old_data.y[node.name])
            node.in_test_mask = bool(old_data.test_mask[node.name])
            node.in_train_mask = bool(old_data.train_mask[node.name])
            node.in_val_mask = bool(old_data.val_mask[node.name])
            node.name = ind

    def __generate_feature_tensor(self) -> Tensor:
        """Generates a tensor of features (known as `x`) from nodes"""
        matrix = torch.zeros((len(self.nodes), len(self.nodes[0].features)))
        for i, node in enumerate(self.nodes):
            print(node.features)
            matrix[i] = node.features

        return matrix

    def __generate_edge_index(self) -> Tensor:
        """Generates a 2xE tensor where all edges are stored"""
        matrix = [[], []]
        for node, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                matrix[0].append(node.name)
                matrix[1].append(neighbor.name)

        return Tensor(matrix)

    def __get_node_fields(self, func_to_apply) -> Tensor:
        """
        Returns a tensor of node fields.
        Example: `__get_node_fields(lambda node: node.label` will return a tensor of all labels
        """
        return Tensor([func_to_apply(node) for node in self.nodes])

    # FIXME
    def __get_train_mask(self):
        res = torch.zeros((1, len(self.nodes)))

        for i, node in enumerate(self.nodes):
            res[i] = bool(node.in_train_mask)
        return res

    # FIXME
    def __get_test_mask(self):
        pass

    # FIXME
    def __get_val_mask(self):
        pass

    def get_graph_data(self, old_data: Data) -> Data:
        self.__rename_nodes(old_data)
        features = self.__generate_feature_tensor()
        edge_index = self.__generate_edge_index()
        labels = Tensor([node.label for node in self.nodes])
        train_mask = Tensor([bool(node.in_train_mask) for node in self.nodes])
        test_mask = Tensor([bool(node.in_test_mask) for node in self.nodes])
        val_mask = Tensor([bool(node.in_val_mask) for node in self.nodes])
        return Data(x=features,
                    edge_index=edge_index,
                    y=labels,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask)


if __name__ == "__main__":
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    graph = Graph(data.edge_index, data.x)
    graph.get_graph_data(data)
