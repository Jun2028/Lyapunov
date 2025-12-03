"""Standalone helpers for manipulating expression trees and generating random equations.

This module mirrors the ``Node`` utilities embedded in ``ode.py`` but keeps them
decoupled from ``ODEEnvironment`` so they can be reused in lightweight demos.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

class DomainError(RuntimeError):
    """Raised when the domain of an expression cannot be determined."""


class Node:
    """
    Minimal symbolic expression tree.
    """

    def __init__(self, value, children=None):
        self.value = value
        self.children: List["Node"] = children if children else []
        self._domain: Optional[List["Node"]] = None

    def push_child(self, child: "Node") -> None: 
        '''
        Same as self.children.append(child)
        '''
        self.children.append(child)

    def prefix(self) -> str:
        """
        Enumerate tree nodes in prefix (Polish) notation.
        """
        s = str(self.value)
        for c in self.children:
            s += ", " + c.prefix()
        return s

    def infix(self) -> str:
        """
        Convert tree into a human readable infix expression.
        """
        nb_children = len(self.children)
        if nb_children <= 1:
            s = str(self.value)
            if isinstance(self.value, int) and self.value < 0:
                s = "(" + s + ")"
            elif nb_children == 1:
                s += "(" + self.children[0].infix() + ")"
            return s
        s = "(" + self.children[0].infix()
        for c in self.children[1:]:
            s = s + " " + str(self.value) + " " + c.infix()
        return s + ")"

    def __len__(self) -> int:
        """
        Return the number of nodes contained in the expression.
        """
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self) -> str:
        return self.infix()

    def __add__(self, node):
        if isinstance(node, int):
            node = Node(node)
        return Node("+", [self, node])

    def __sub__(self, node):
        if isinstance(node, int):
            node = Node(node)
        return Node("-", [self, node])

    def __radd__(self, node):
        if isinstance(node, int):
            node = Node(node)
        return Node("+", [node, self])

    def __mul__(self, node):
        if isinstance(node, int):
            node = Node(node)
        return Node("*", [self, node])

    def __rmul__(self, node):
        if isinstance(node, int):
            node = Node(node)
        return Node("*", [node, self])

    def __ne__(self, node):
        return Node("!=", [self, autocast(node)])

    def __le__(self, node):
        return Node("<=", [self, autocast(node)])

    def __lt__(self, node):
        return Node("<", [self, autocast(node)])

    def __ge__(self, node):
        return Node("<=", [autocast(node), self])

    def __gt__(self, node):
        return Node("<", [autocast(node), self])

    def __pow__(self, node) -> "Node":
        return Node("^", [self, autocast(node)])

    def ln(self) -> "Node":
        return Node("ln", [self])

    def exp(self) -> "Node":
        return Node("exp", [self])

    def eq(self, node) -> bool:
        return self.prefix() == node.prefix()

    def clone(self) -> "Node":
        return Node(self.value, [c.clone() for c in self.children])

    def replace(self, x: "Node", y: "Node") -> "Node":
        if self.eq(x):
            return y
        return Node(self.value, [c.replace(x, y) for c in self.children])

    def replace_ops(self, ops1: str, ops_lst: List[str], except_exp: bool = False) -> "Node":
        if self.value == ops1:
            if except_exp and len(self.children) == 1 and self.children[0].value == "exp":
                return Node(self.value, [c.clone() for c in self.children])
            current_children = [c.clone() for c in self.children]
            for op in ops_lst:
                new_node = Node(op, current_children)
                current_children = [new_node]
            return new_node
        return Node(self.value, [c.replace_ops(ops1, ops_lst, except_exp) for c in self.children])

    def remove_ops(
        self, ops: str, parent_node: Optional["Node"] = None, node_index: Optional[int] = 0
    ) -> "Node":
        if self.value == ops:
            assert len(self.children) == 1
            if parent_node is None:
                return self.children[0].clone()
            new_children = [child.clone() for child in parent_node.children]
            new_children[node_index] = self.children[0].clone()
            return Node(parent_node.value, new_children)
        return Node(self.value, [c.remove_ops(ops, self, i) for i, c in enumerate(self.children)])

    def _find_domain(self, refresh: bool = False) -> None:
        self._domain = []
        for c in self.children:
            if refresh or c.domain() is None:
                c._find_domain(refresh)
            self._domain.extend(c.domain())
        if self.value in {"acos", "asin"}:
            self._domain.append(self.children[0] + Node(1) >= 0)
            self._domain.append(Node(1) - self.children[0] >= 0)
        if self.value == "tan":
            self._domain.append(self.children[0] + Node("/", [Node("pi"), Node(2)]) > 0)
            self._domain.append(Node("/", [Node("pi"), Node(2)]) - self.children[0] > 0)
        if self.value == "sqrt":
            self._domain.append(self.children[0] >= 0)
        elif self.value == "ln":
            self._domain.append(self.children[0] > 0)
        elif self.value == "div":
            self._domain.append(self.children[1] != 0)
        elif self.value == "^":
            assert len(self.children) == 2
            c0, c1 = self.children
            c2 = c1.value
            try:
                c2_f = float(c2)
                c_is_int = int(c2_f) - c2_f == 0
                if c2_f < 0 and c_is_int:
                    self._domain.append(c0 != 0)
                elif c2_f < 0:
                    self._domain.append(c0 > 0)
            except ValueError as e:
                if e.args[0].startswith("could not convert string to float:"):
                    self._domain.append(c0 > 0)
                else:
                    raise DomainError(f"Domain not implemented for exponent {c2}, current node is {self}")

    def domain(self, refresh: bool = False) -> List["Node"]:
        if (self._domain is None) or refresh:
            self._find_domain()
        return self._domain
    
    def to_graphviz(self, name: str = "Expression", node_shape: str = "circle", **graph_kwargs):
        """
        Return a Graphviz Digraph rendering the tree.
        """
        dot = Digraph(name=name, **graph_kwargs)
        if node_shape:
            dot.attr("node", shape=node_shape)
        counter = itertools.count()
        def add(node: "Node") -> str:
            node_id = f"n{next(counter)}"
            dot.node(node_id, label=str(node.value))
            for child in node.children:
                child_id = add(child)
                dot.edge(node_id, child_id)
            return node_id
        add(self)
        return dot



def autocast(x):
    if isinstance(x, Node):
        return x
    if isinstance(x, int):
        return Node(x)
    raise RuntimeError(f"Unexpected type: {x}")


@dataclass
class RandomExpressionGenerator:
    """
    Lightweight wrapper around ``generate_tree`` for demo usage.
    """

    binaries: Sequence[str] = ("+", "-", "*") #can be expanded by just adding in other operators.
    unaries: Sequence[str] = ("sin", "cos", "exp")
    variables: Optional[Sequence[str]] = None
    prob_int: float = 0.4
    max_int: int = 5
    positive_ints: bool = False
    non_zero_only: bool = False
    max_ops: int = 32
    allow_unary_nodes: bool = True
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if not self.binaries:
            raise ValueError("At least one binary operator is required.")
        if self.max_ops <= 0:
            raise ValueError("max_ops must be positive.")
        self.binaries = tuple(self.binaries)
        self.unaries = tuple(self.unaries)
        self._variables = tuple(self.variables) if self.variables else tuple()
        self.rng = self.rng if self.rng is not None else np.random.default_rng()
        self.unary = self.allow_unary_nodes and bool(self.unaries)
        self.distrib = self._generate_dist(2 * self.max_ops)

    def generate_tree(self, nb_ops: int, degree: int, index: int = 0) -> Node:
        """
        Build a random expression tree containing ``nb_ops`` operator nodes. This nb_ops is just n in the algorithm.
        """
        if nb_ops < 0:
            raise ValueError("nb_ops must be non-negative.")
        if nb_ops > self.max_ops:
            raise ValueError(f"nb_ops={nb_ops} exceeds pre-computed max_ops={self.max_ops}.")
        if degree < 1:
            raise ValueError("degree must be at least 1.")

        tree = Node(0)
        empty_nodes: List[Node] = [tree]
        next_en = 0
        nb_empty = 1
        remaining_ops = nb_ops

        while remaining_ops > 0:
            next_pos, arity = self._sample_next_pos(nb_empty, remaining_ops)
            for node in empty_nodes[next_en : next_en + next_pos]:
                node.value = self._generate_leaf(degree, index)
            next_en += next_pos
            empty_nodes[next_en].value = self._generate_ops(arity)
            for _ in range(arity):
                child = Node(0)
                empty_nodes[next_en].push_child(child)
                empty_nodes.append(child)
            nb_empty += arity - 1 - next_pos
            remaining_ops -= 1
            next_en += 1

        for node in empty_nodes[next_en:]:
            node.value = self._generate_leaf(degree, index)
        return tree

    # --- helpers -----------------------------------------------------------------
    def _variable_name(self, idx: int) -> str:
        if idx < len(self._variables):
            return self._variables[idx]
        return f"x{idx}"

    def _draw_integer(self) -> int:
        max_int = max(1, self.max_int)
        if self.positive_ints:
            low = 1 if self.non_zero_only else 0
            return int(self.rng.integers(low, max_int + 1))
        if self.non_zero_only:
            value = 0
            while value == 0:
                value = int(self.rng.integers(-max_int, max_int + 1))
            return value
        return int(self.rng.integers(-max_int, max_int + 1))

    def _generate_leaf(self, degree: int, index: int):
        if self.prob_int and self.rng.random() < self.prob_int:
            return self._draw_integer()
        target_index = index if degree == 1 else int(self.rng.integers(0, degree))
        return self._variable_name(target_index)

    def _generate_ops(self, arity: int) -> str:
        if arity == 1:
            if not self.unary:
                raise RuntimeError("Unary arity requested but no unary operators configured.")
            return str(self.rng.choice(self.unaries))
        return str(self.rng.choice(self.binaries))

    def _generate_dist(self, max_ops: int) -> List[List[int]]:
        p1 = 1 if self.unary else 0
        D: List[List[int]] = []
        D.append([0] + [1 for _ in range(1, 2 * max_ops + 1)])
        for n in range(1, 2 * max_ops + 1):
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        return D

    def _sample_next_pos(self, nb_empty: int, nb_ops: int) -> Tuple[int, int]:
        if nb_empty <= 0:
            raise ValueError("nb_empty must stay positive.")
        if nb_ops <= 0 or nb_ops >= len(self.distrib):
            raise ValueError("nb_ops is out of bounds for the precomputed distribution.")
        probs = []
        if self.unary:
            for i in range(nb_empty):
                probs.append(self.distrib[nb_ops - 1][nb_empty - i])
        for i in range(nb_empty):
            probs.append(self.distrib[nb_ops - 1][nb_empty - i + 1])
        total = self.distrib[nb_ops][nb_empty]
        if total == 0:
            raise RuntimeError("Tree enumeration distribution is zero; increase max_ops.")
        probs = np.array([p / total for p in probs], dtype=np.float64)
        choice = int(self.rng.choice(len(probs), p=probs))
        arity = 1 if self.unary and choice < nb_empty else 2
        choice %= nb_empty
        return choice, arity
