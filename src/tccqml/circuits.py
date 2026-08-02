"""Circuitos e embeddings quânticos usados no TCC."""

import pennylane as qml
from pennylane import numpy as pnp


def angle_embedding_circuit(n_qubits: int = 2):
    """Retorna um QNode que codifica x via rotações RY e mede <Z_0>.

    Serve como smoke test da stack: se isto roda, PennyLane +
    diferenciação automática estão funcionando.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def random_weights(n_layers: int, n_qubits: int, seed: int = 42):
    """Pesos iniciais reprodutíveis para BasicEntanglerLayers."""
    rng = pnp.random.default_rng(seed)
    return pnp.array(rng.uniform(0, 2 * pnp.pi, size=(n_layers, n_qubits)), requires_grad=True)
