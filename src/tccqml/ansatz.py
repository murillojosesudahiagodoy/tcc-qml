"""Ansatz variacional (Etapa 2).

Este módulo fica CONGELADO a partir da Etapa 8. Toda a comparação da Etapa 9
depende de o ansatz ser idêntico entre codificações — se ele mudar junto,
não dá para atribuir a diferença de desempenho à codificação.
"""

import pennylane as qml
from pennylane import numpy as pnp


def strongly_entangling(weights, wires) -> None:
    """Camadas de rotação genérica (Rot) seguidas de anel de CNOTs."""
    qml.StronglyEntanglingLayers(weights, wires=wires)


def weights_shape(n_layers: int, n_qubits: int) -> tuple[int, ...]:
    return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)


def init_weights(n_layers: int, n_qubits: int, seed: int = 42, escala: float = 0.1):
    """Pesos iniciais pequenos e reprodutíveis.

    A escala pequena é deliberada: inicialização uniforme em [0, 2pi] num
    circuito profundo cai em barren plateau, onde o gradiente é praticamente
    nulo e o treino não sai do lugar (Etapa 5).
    """
    forma = weights_shape(n_layers, n_qubits)
    rng = pnp.random.default_rng(seed)
    return pnp.array(rng.normal(0.0, escala, size=forma), requires_grad=True)