"""Classificador quântico variacional (Etapas 3 e 8).

Monta a cadeia do roteiro:

    x -> psi(x) -> U(theta) psi(x) -> <Z_0> -> y_chapeu
"""

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from tccqml.ansatz import init_weights, strongly_entangling, weights_shape
from tccqml.embeddings import get_encoding


@dataclass
class Classificador:
    """Circuito + metadados necessários para treinar e para medir custo."""

    circuit: object
    encoding: str
    n_qubits: int
    n_layers: int
    weights_shape: tuple

    @property
    def n_params(self) -> int:
        """Parâmetros treináveis do ansatz (+1 do bias clássico)."""
        return int(pnp.prod(pnp.array(self.weights_shape))) + 1


def build(
    encoding: str,
    n_features: int,
    n_layers: int = 2,
    device: str = "default.qubit",
) -> Classificador:
    """Constrói o QNode que devolve <Z_0> em [-1, 1]."""
    enc = get_encoding(encoding)
    n_qubits = enc.n_qubits(n_features)
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, weights):
        enc.apply(x, wires=range(n_qubits))
        strongly_entangling(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return Classificador(
        circuit=circuit,
        encoding=encoding,
        n_qubits=n_qubits,
        n_layers=n_layers,
        weights_shape=weights_shape(n_layers, n_qubits),
    )


def pesos_iniciais(clf: Classificador, seed: int = 42):
    """Devolve (weights, bias) prontos para o otimizador."""
    w = init_weights(clf.n_layers, clf.n_qubits, seed=seed)
    b = pnp.array(0.0, requires_grad=True)
    return w, b


def saida_continua(clf: Classificador, weights, bias, X):
    """<Z_0> + bias para um lote de amostras. Imagem aproximada: [-1, 1]."""
    return clf.circuit(X, weights) + bias


def prever(clf: Classificador, weights, bias, X):
    """Converte a saída contínua em rótulo {0, 1}."""
    z = np.asarray(saida_continua(clf, weights, bias, X), dtype=float)
    return (z > 0).astype(int)