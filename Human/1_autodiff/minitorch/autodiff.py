from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    less = f(*(vals[:arg] + (vals[arg] - epsilon,) + vals[arg + 1:]))
    more = f(*(vals[:arg] + (vals[arg] + epsilon,) + vals[arg + 1:]))
    return (more - less) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    topo, visited = [], set()
    def build(var: Variable) -> None:
        if var.unique_id not in visited:
            visited.add(var.unique_id)
            for parent in var.parents:
                build(parent)
            if not var.is_constant():
                topo.append(var)
    build(variable)
    return topo[::-1]
        

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    order = topological_sort(variable)
    derivative_dic = {}
    derivative_dic[variable.unique_id] = deriv
    
    for var in order:
        if var.is_constant() or var.is_leaf():
            continue
        
        d_output = derivative_dic[var.unique_id]
        
        for v_input, d_input in var.chain_rule(d_output):
            if v_input.is_constant():
                continue
            if v_input.unique_id not in derivative_dic.keys():
                derivative_dic[v_input.unique_id] = 0.0
            derivative_dic[v_input.unique_id] += d_input
            
    for var in order:
        if var.is_leaf():
            var.accumulate_derivative(derivative_dic[var.unique_id])
        

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
