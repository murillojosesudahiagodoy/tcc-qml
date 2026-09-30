"""Ansatz variacional (Etapa 2).

Este módulo fica CONGELADO a partir da Etapa 8. Toda a comparação da Etapa 9
depende de o ansatz ser idêntico entre codificações — se ele mudar junto,
não dá para atribuir a diferença de desempenho à codificação.

A única exceção deliberada é o ansatz ``local``: uma variante SEM portas
de dois qubits, usada exclusivamente na ablação que testa a Previsão 3
(Eq. 2.83). Ele nunca entra na comparação principal.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pennylane as qml
from pennylane import numpy as pnp


@dataclass(frozen=True)
class Ansatz:
    """Um circuito treinável U(theta), com a forma dos seus pesos."""

    name: str
    apply: Callable[..., None]
    weights_shape: Callable[[int, int], tuple[int, ...]]
    descricao: str = ""


def strongly_entangling(weights, wires) -> None:
    """Camadas de rotação genérica (Rot) seguidas de anel de CNOTs."""
    qml.StronglyEntanglingLayers(weights, wires=wires)


def weights_shape(n_layers: int, n_qubits: int) -> tuple[int, ...]:
    return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)


def local(weights, wires) -> None:
    """Ablação: rotações genéricas por qubit, SEM nenhuma CNOT.

    U(theta) = prod_i Rot(theta_i), ou seja, um produto tensorial de unitárias
    de um qubit. Tem exatamente o mesmo número de parâmetros que o
    ``strongly_entangling`` (3Ln) — a única coisa removida é o entrelaçamento,
    o que isola o seu efeito.

    Com estado produto na entrada e U = u_0 ⊗ u_1, a saída <Z_0> colapsa numa
    função só de x_1 (Eq. 2.83): o segundo atributo é codificado, ocupa um
    qubit e é completamente ignorado pela medição.
    """
    wires = list(wires)
    n_layers = pnp.shape(weights)[0]
    for camada in range(n_layers):
        for i, w in enumerate(wires):
            qml.Rot(
                weights[camada, i, 0],
                weights[camada, i, 1],
                weights[camada, i, 2],
                wires=w,
            )


def weights_shape_local(n_layers: int, n_qubits: int) -> tuple[int, ...]:
    """Mesma forma do StronglyEntanglingLayers: (L, n, 3)."""
    return (n_layers, n_qubits, 3)


ANSATZE = {
    "strongly_entangling": Ansatz(
        name="strongly_entangling",
        apply=strongly_entangling,
        weights_shape=weights_shape,
        descricao="Rot por qubit + anel de CNOTs (padrão do trabalho, congelado)",
    ),
    "local": Ansatz(
        name="local",
        apply=local,
        weights_shape=weights_shape_local,
        descricao="Rot por qubit, sem CNOTs — só para a ablação da Previsão 3",
    ),
}


def get_ansatz(name: str) -> Ansatz:
    if name not in ANSATZE:
        raise ValueError(f"ansatz desconhecido: {name!r}. Opções: {sorted(ANSATZE)}")
    return ANSATZE[name]


def init_weights(
    n_layers: int,
    n_qubits: int,
    seed: int = 42,
    escala: float = 0.1,
    forma: tuple[int, ...] | None = None,
):
    """Pesos iniciais pequenos e reprodutíveis.

    A escala pequena é deliberada: inicialização uniforme em [0, 2pi] num
    circuito profundo cai em barren plateau, onde o gradiente é praticamente
    nulo e o treino não sai do lugar.

    ``forma`` permite inicializar um tensor de forma arbitrária (o
    re-uploading tem os pesos ditados pela codificação, não por (L, n)).
    """
    if forma is None:
        forma = weights_shape(n_layers, n_qubits)
    rng = pnp.random.default_rng(seed)
    return pnp.array(rng.normal(0.0, escala, size=forma), requires_grad=True)
