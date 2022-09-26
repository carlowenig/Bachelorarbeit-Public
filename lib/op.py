from __future__ import annotations
import functools
import numpy as np
import numpy.typing as npt
import sympy
import scipy

from .state import create_basis_state, standardize_phi


def hconj(matrix: npt.NDArray[np.complex128]):
    return matrix.conj().T


class Op:
    name = None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return state

    @property
    def dagger(self) -> Op:
        return self

    def __mul__(self, other: Op | np.ndarray | complex | float | int):
        if isinstance(other, Op):
            return ChainedOp(self, other)
        elif isinstance(other, np.ndarray):
            assert (
                len(other.shape) == 1
            ), "Can only multiply with column vectors from the left (kets)."
            return self(other)
        else:
            return LinearCombinationOp({self: other})

    def __rmul__(self, other: Op | np.ndarray | complex | float | int):
        if isinstance(other, Op):
            return ChainedOp(other, self)
        elif isinstance(other, np.ndarray):
            assert (
                other.shape[0] == 1
            ), "Can only multiply with row vectors from the right (bras)."
            state_dagger = hconj(other)
            return hconj(self.dagger(state_dagger))
        else:
            return LinearCombinationOp({self: other})

    def __truediv__(self, other: complex | float | int):
        return LinearCombinationOp({self: 1 / other})

    def __add__(self, other):
        return LinearCombinationOp({self: 1, as_op(other): 1})

    def __neg__(self):
        return LinearCombinationOp({self: -1})

    def __radd__(self, other: Op | complex | float | int):
        return self + other

    def __sub__(self, other: Op | complex | float | int):
        return self + (-other)

    def __rsub__(self, other: Op | complex | float | int):
        return -self + other

    def __pow__(self, power: int):
        return PowerOp(self, power)

    def expval(self, state: np.ndarray) -> complex:
        return hconj(state) @ self(state)

    @functools.cache
    def compute_matrix(self, N: int) -> np.ndarray:
        matrix = np.zeros((N, N), dtype=np.complex128)
        for k in range(N):
            basis_state = create_basis_state(k, N)
            result_ket = self(basis_state)
            matrix[:, k] = result_ket
        return matrix

    def compute_sympy_matrix(self, N: int) -> sympy.Matrix:
        return sympy.Matrix(self.compute_matrix(N))

    def isclose(self, other: Op, test_N=100) -> bool:
        return np.allclose(self.compute_matrix(test_N), other.compute_matrix(test_N))

    @functools.cache
    def is_hermitian(self, test_N=100) -> bool:
        return self.isclose(self.dagger, test_N=test_N)

    @functools.cache
    def is_unitary(self, test_N=100) -> bool:
        return (self.dagger * self).isclose(identity_op, test_N=test_N)

    @functools.cache
    def is_real(self, test_N=100) -> bool:
        return np.isreal(self.compute_matrix(test_N)).all()

    def eig(self, N, method=np.linalg.eig):
        eigvals, eigstates_T = method(self.compute_matrix(N))
        eigstates = np.array([standardize_phi(eigstate) for eigstate in eigstates_T.T])
        return eigvals, eigstates

    def eigh(self, N, validate=True):
        if validate:
            assert self.is_hermitian(N), "Operator is not Hermitian."
        return self.eig(N, method=np.linalg.eigh)

    def validate_dagger(self, test_N=100):
        matrix = self.compute_matrix(test_N)
        dagger_matrix = self.dagger.compute_matrix(test_N)
        if not np.allclose(dagger_matrix, matrix.conj().T):
            raise ValueError("Dagger is not valid.")

    def __str__(self):
        return self.name or super().__str__()


identity_op = Op()
identity_op.name = "I"


def commutator(a: Op, b: Op):
    return a * b - b * a


class ChainedOp(Op):
    def __init__(self, *ops: Op):
        self.ops = []
        for op in ops:
            if isinstance(op, ChainedOp):
                self.ops.extend(op.ops)
            else:
                self.ops.append(op)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        for op in reversed(self.ops):
            state = op(state)

        return state

    @property
    def dagger(self) -> Op:
        return ChainedOp(*reversed([op.dagger for op in self.ops]))

    @functools.cache
    def compute_matrix(self, N: int) -> np.ndarray:
        matrix = np.identity(N, dtype=np.complex128)
        for op in reversed(self.ops):
            matrix = op.compute_matrix(N) @ matrix
        return matrix

    def __str__(self):
        return " ".join(str(op) for op in self.ops)


class PowerOp(Op):
    def __init__(self, op: Op, power: int):
        self.op = op
        assert (
            isinstance(power, int) and power >= 0
        ), "Power must be a non-negative integer."
        self.power = power

    def __call__(self, state: np.ndarray) -> np.ndarray:
        for _ in range(self.power):
            state = self.op(state)
        return state

    @property
    def dagger(self) -> Op:
        return PowerOp(self.op.dagger, self.power)

    @functools.cache
    def compute_matrix(self, N: int) -> np.ndarray:
        matrix = np.linalg.matrix_power(self.op.compute_matrix(N), self.power)
        return matrix

    def is_hermitian(self, test_N) -> bool:
        return self.op.is_hermitian(test_N)

    def __str__(self):
        return f"{self.op}^{self.power}"


class LinearCombinationOp(Op):
    def __init__(self, terms: dict[Op, complex]):
        # self.terms = {}
        # for op, coeff in terms.items():
        #     if coeff == 0:
        #         continue
        #     elif isinstance(op, LinearCombinationOp):
        #         for op2, coeff2 in op.terms.items():
        #             self.terms[op2] = complex(terms.get(op2, 0) + coeff * coeff2)
        #     else:
        #         self.terms[op] = complex(coeff)
        self.terms = terms
        # self._simplify()

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return sum(coeff * op(state) for op, coeff in self.terms.items())

    @property
    def dagger(self) -> Op:
        return LinearCombinationOp(
            {op.dagger: coeff.conjugate() for op, coeff in self.terms.items()}
        )

    @functools.cache
    def compute_matrix(self, N: int) -> np.ndarray:
        return sum(coeff * op.compute_matrix(N) for op, coeff in self.terms.items())

    # def _simplify(self):
    #     terms = {}
    #     for op, coeff in self.terms.items():
    #         if coeff == 0:
    #             continue
    #         elif isinstance(op, LinearCombinationOp):
    #             for op2, coeff2 in op.terms.items():
    #                 terms[op2] = complex(terms.get(op2, 0) + coeff * coeff2)
    #         else:
    #             terms[op] = complex(coeff)
    #     self.terms = terms

    def __str__(self):
        return (
            "("
            + " + ".join(f"{coeff} * {op}" for op, coeff in self.terms.items())
            + ")"
        )


zero_op = LinearCombinationOp({})


class MatrixOp(Op):
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        assert self.matrix.shape[0] == self.matrix.shape[1]

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if len(state) != self.matrix.shape[0]:
            raise ValueError("Dimension mismatch.")
        return self.matrix @ state

    @property
    def dagger(self) -> Op:
        return MatrixOp(self.matrix.conj().T)

    def __repr__(self):
        return f"MatrixOp({self.matrix})"

    def __eq__(self, other):
        return isinstance(other, MatrixOp) and np.all(self.matrix == other.matrix)

    def __hash__(self):
        return hash(self.matrix.tobytes())

    def compute_matrix(self, N: int) -> np.ndarray:
        if N != self.matrix.shape[0]:
            raise ValueError("Dimension mismatch.")
        return self.matrix

    def __str__(self):
        return f"Mat({self.matrix.shape[0]})"


class FunctionOp(Op):
    def __init__(self, function, dagger_function=None):
        self.function = function
        self.dagger_function = dagger_function or function

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.function(state)

    @property
    def dagger(self) -> Op:
        return FunctionOp(self.dagger_function, self.function)

    def __repr__(self):
        return f"FunctionOp({self.function}, {self.dagger_function})"

    def __eq__(self, other):
        return (
            isinstance(other, FunctionOp)
            and self.function == other.function
            and self.dagger_function == other.dagger_function
        )

    def __hash__(self):
        return hash((self.function, self.dagger_function))

    @property
    def is_hermitian(self) -> bool:
        return self.function == self.dagger_function or super().is_hermitian

    def __str__(self):
        return f"Func({self.function.__name__})"


class BasisFunctionOp(Op):
    def __init__(self, function, dagger_function=None):
        self.function = function
        self.dagger_function = dagger_function or function

    def __call__(self, state: np.ndarray) -> np.ndarray:
        sum = np.zeros_like(state)
        N = len(state)
        for k in range(N):
            sum += self.function(k, N) * state[k]
        return sum

    @property
    def dagger(self) -> Op:
        return BasisFunctionOp(self.dagger_function, self.function)

    def __repr__(self):
        return f"BasisFunctionOp({self.function}, {self.dagger_function})"

    def __eq__(self, other):
        return (
            isinstance(other, BasisFunctionOp)
            and self.function == other.function
            and self.dagger_function == other.dagger_function
        )

    def __hash__(self):
        return hash((self.function, self.dagger_function))

    @functools.cache
    def compute_matrix(self, N: int) -> np.ndarray:
        matrix = np.zeros((N, N), dtype=complex)
        for k in range(N):
            # Set k-th column
            matrix[:, k] = self.function(k, N)
        return matrix

    @property
    def is_hermitian(self) -> bool:
        return self.function == self.dagger_function or super().is_hermitian

    def __str__(self):
        return f"BasisFunc({self.function.__name__})"


class ExpOp(Op):
    def __init__(self, op: Op):
        self.op = op

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.compute_matrix(len(state)) @ state
        # result = np.zeros_like(state)
        # for k in range(self.order):
        #     summand = state
        #     for j in range(k):
        #         summand = self.op(summand)
        #     summand /= math.factorial(k)
        #     result += summand
        # return result

    @property
    def dagger(self) -> Op:
        return ExpOp(self.op.dagger)

    def __repr__(self):
        return f"ExpOp({self.op})"

    def __eq__(self, other):
        return isinstance(other, ExpOp) and self.op == other.op

    def __hash__(self):
        return hash(self.op)

    @functools.cache
    def compute_matrix(self, N: int) -> np.ndarray:
        return scipy.linalg.expm(self.op.compute_matrix(N))

    def __str__(self):
        return f"exp({self.op})"


def as_op(input):
    if isinstance(input, Op):
        return input
    elif isinstance(input, np.ndarray):
        return MatrixOp(input)
    elif callable(input):
        return FunctionOp(input)
    elif input == 0:
        return zero_op
    elif isinstance(input, (complex, float, int)):
        return identity_op * input
    else:
        raise ValueError(f"Cannot create operator from input {input}.")
