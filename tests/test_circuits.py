"""Testes do módulo de circuitos."""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from tccqml.circuits import angle_embedding_circuit, random_weights


def test_circuito_retorna_valor_esperado_valido():
    circuit = angle_embedding_circuit(n_qubits=2)
    x = pnp.array([0.1, 0.2], requires_grad=False)
    w = random_weights(n_layers=1, n_qubits=2)

    out = circuit(x, w)

    assert -1.0 <= float(out) <= 1.0


def test_gradiente_existe_e_nao_e_nulo():
    """Se o autodiff quebrar, todo o treinamento variacional quebra."""
    circuit = angle_embedding_circuit(n_qubits=2)
    x = pnp.array([0.5, 0.9], requires_grad=False)
    w = random_weights(n_layers=2, n_qubits=2)

    grad = qml.grad(circuit, argnums=1)(x, w)

    assert grad.shape == (2, 2)
    assert np.any(np.abs(grad) > 1e-8)


def test_reprodutibilidade_dos_pesos():
    assert np.allclose(random_weights(2, 3, seed=7), random_weights(2, 3, seed=7))
