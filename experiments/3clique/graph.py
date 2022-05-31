from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import BoolTensor, LongTensor, Tensor
from torch_geometric.data import Data


class Node:
    """Representation of a node from dataset"""

    name: int
    features: Tensor
    label: int
    in_train_mask: bool
    in_test_mask: bool
    in_val_mask: bool

    def __init__(self, name: int, features: Tensor):
        """Create a new node by label and it's feature vector

        Args:
            name (int): Node's label (and index)
            features (Tensor): 1xF feature vector
        """
        self.name = name
        self.features = features

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return f"Node(name={self.name})"

    def __repr__(self):
        return str(self)


class Graph:
    """Representation of dataset"""

    nodes: List[Node]
    adjacency_list: Dict[Node, Set[Node]]

    def __init__(self, data: Data):
        """Creates a graph from a piece of dataset

        Args:
            data (Data): Dataset
        """
        self.nodes = []
        self.adjacency_list = {}

        nodes_num = len(data.edge_index[0])

        for i in range(nodes_num):
            # find nodes associated with numbers
            source = self.__find_node_or_create(int(data.edge_index[0][i]), data.x)
            source.in_val_mask = data.val_mask[source.name]
            destination = self.__find_node_or_create(int(data.edge_index[1][i]), data.x)
            destination.in_val_mask = data.val_mask[destination.name]

            # add edges to `adjacency_list`
            self.__add_edge(source, destination)
            self.__add_edge(source=destination, destination=source)

    def __add_edge(self, source: Node, destination: Node):
        """Adds an edge (source, destination) to adjacency list"""
        # if key doesn't exist, create one with default value
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
        return None

    def edge_exists(self, source: Node, destination: Node) -> bool:
        """Returns `True` if edge between source and destination exists

        Args:
            source (Node): Source node
            destination (Node): Target node

        Returns:
            bool: True if edge (source, destination) exists
        """
        return destination in self.adjacency_list[source]

    def nodes_form_3clique(self, a: Node, b: Node, c: Node) -> bool:
        """Returns True if 3 given nodes form a triangle (3-clique)

        Args:
            a (Node): Some node
            b (Node): Some node
            c (Node): Some node

        Returns:
            bool: True if all of given nodes has an edge between them
        """
        a_b = self.edge_exists(a, b)
        a_c = self.edge_exists(a, c)
        b_c = self.edge_exists(b, c)
        return a_b and a_c and b_c

    def get_all_3cliques(self) -> List[Tuple[Node, Node, Node]]:
        """Returns a list of 3-cliques.
        One node might belong to multiple 3-cliques.

        Returns:
            List[Tuple[Node, Node, Node]]: List of (Node, Node, Node)
        """
        result = set()

        for a in self.nodes:
            for b in self.adjacency_list[a]:
                common_neighbors = self.adjacency_list[a] & self.adjacency_list[b]

                for c in common_neighbors:
                    # Sort in order to remove duplicate cliques (a, b, c) is same as (c, a, b) and etc
                    clique = [a, b, c]
                    clique.sort()
                    result.add(tuple(clique))

        return list(result)

    def __cleanup_adj_list(self):
        extra_nodes = set(self.adjacency_list.keys()) - set(self.nodes)
        for node in extra_nodes:
            del self.adjacency_list[node]

    def replace_3clique_with_node(self, clique: Tuple[Node, Node, Node]):
        """Merges nodes from a 3-clique into one node sharing their features

        Args:
            clique (Tuple[Node, Node, Node]): 3 Nodes forming a triangle
        """
        a, b, c = clique

        neighbors = (
            self.adjacency_list[a] | self.adjacency_list[b] | self.adjacency_list[c]
        )
        neighbors.remove(a)
        neighbors.remove(b)
        neighbors.remove(c)

        # remove all edges coming to these nodes
        for source in clique:
            for dest in self.adjacency_list[source]:
                self.adjacency_list[dest] -= set(clique)
            self.adjacency_list.pop(source)
            self.nodes.remove(source)

        # for source in clique:
        #     for destination in self.adjacency_list[source]:
        #         self.adjacency_list[destination].remove(source)
        #     self.adjacency_list.pop(source)
        #     self.nodes.remove(source)

        # crate a "general" node
        common_features: Tensor = (a.features + b.features + c.features) / 3
        new_node_name = a.name
        common_node = Node(new_node_name, common_features)
        self.nodes.append(common_node)
        self.adjacency_list[common_node] = set()

        for neighbor in neighbors:
            self.__add_edge(common_node, neighbor)
            self.__add_edge(neighbor, common_node)

    def __rename_nodes(self, old_data: Data):
        """Generates new names for all the nodes

        Args:
            old_data (Data): _description_
        """
        # Dict [old_name, new_name]
        names = {}
        self.nodes.sort()
        for ind, node in enumerate(self.nodes):
            node.label = int(old_data.y[node.name])
            node.in_test_mask = bool(old_data.test_mask[node.name])
            node.in_train_mask = bool(old_data.train_mask[node.name])
            node.in_val_mask = bool(old_data.val_mask[node.name])
            names[node.name] = ind
            node.name = ind

        # Rebuild adjacency list
        self.adjacency_list = {}

        for nd, nbr in zip(*old_data.edge_index):
            # fr, to = None, None
            try:
                fr = self.__find_node_by_name(names[int(nd)])
                to = self.__find_node_by_name(names[int(nbr)])
            except KeyError:
                continue
            if fr not in self.adjacency_list.keys():
                self.adjacency_list[fr] = set()

            self.adjacency_list[fr].add(to)

    def __generate_feature_tensor(self) -> Tensor:
        """Generates a tensor of features (known as `x`) from nodes"""
        matrix = []
        for node in self.nodes:
            matrix.append(list(node.features))

        return Tensor(matrix)

    def __get_number_of_edges(self) -> int:
        ans = 0
        for neighbors in self.adjacency_list.values():
            ans += len(neighbors)
        return ans

    def __generate_edge_index(self) -> Tensor:
        """Generates a 2xE tensor where all edges are stored"""
        # matrix: Tuple[List[int], List[int]] = ([], [])

        matrix = LongTensor(2, self.__get_number_of_edges())
        current_col = 0

        for node, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                matrix[0][current_col] = int(node.name)
                matrix[1][current_col] = int(neighbor.name)
                current_col += 1

        return matrix

    def __get_node_fields(self, func_to_apply) -> Tensor:
        """
        Returns a tensor of node fields.
        Example:
        `__get_node_fields(lambda node: node.label` will return a tensor of all labels
        """
        return Tensor([func_to_apply(node) for node in self.nodes])

    # FIXME
    def __get_train_mask(self):
        res = torch.zeros((1, len(self.nodes)))

        for i, node in enumerate(self.nodes):
            res[i] = bool(node.in_train_mask)
        return res

    def get_graph_data(self, old_data: Data) -> Data:
        """Returns the `Data` (x, y, and all the boolean masks)
        created after performing manipulations on nodes

        Args:
            old_data (Data): the data of nodes before the transformation

        Returns:
            Data: new feature vector, answer vector and boolean masks
        """
        self.__rename_nodes(old_data)
        features = self.__generate_feature_tensor()  # [ [float] ]      (NxF matrix)
        edge_index = self.__generate_edge_index()  # [ [int], [int] ] (2xN matrix)
        labels = LongTensor(
            [int(node.label) for node in self.nodes]
        )  # [int]            (N vector)
        train_mask = BoolTensor(
            [bool(node.in_train_mask) for node in self.nodes]
        )  # [bool]           (N vector)
        test_mask = BoolTensor(
            [bool(node.in_test_mask) for node in self.nodes]
        )  # [bool]           (N vector)
        val_mask = BoolTensor(
            [bool(node.in_val_mask) for node in self.nodes]
        )  # [bool]           (N vector)
        return Data(
            x=features,
            edge_index=edge_index,
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
