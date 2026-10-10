"""Ansatz variacional.

Este módulo fica CONGELADO: toda a comparação entre codificações depende de o
ansatz ser idêntico entre elas — se ele mudar junto, não dá para atribuir a
diferença de desempenho à codificação.

A única exceção deliberada é o ansatz ``local``: uma variante SEM portas
de dois qubits, usada exclusivamente na ablação que testa a hipótese H3
(saída sem entrelaçamento, Cap. 2, "Entrelaçamento e termos cruzados"). Ele
nunca entra na comparação principal.

Com dois qubits e leitura de Z_0, o par de CNOTs que encerra cada camada do
``strongly_entangling`` leva Z_0 a Z_1 (CNOT_{1->0} CNOT_{0->1}, na imagem de
Heisenberg). Por isso a ``Rot`` da última camada no qubit 0 não afeta a saída:
3 dos 12 parâmetros da configuração de referência têm derivada identicamente
nula. Eles continuam contados em `p`, porque fazem parte do circuito e seriam
avaliados pelo parameter-shift (ver `test_rot_final_do_qubit_0_e_inerte`).
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


def weights_shape(L_var: int, n_qubits: int) -> tuple[int, ...]:
    return qml.StronglyEntanglingLayers.shape(n_layers=L_var, n_wires=n_qubits)


def local(weights, wires) -> None:
    """Ablação: rotações genéricas por qubit, SEM nenhuma CNOT.

    U(theta) = prod_i Rot(theta_i), ou seja, um produto tensorial de unitárias
    de um qubit. Tem exatamente o mesmo número de parâmetros que o
    ``strongly_entangling`` (3Ln) — a única coisa removida é o entrelaçamento,
    o que isola o seu efeito.

    Com estado produto na entrada e U = u_0 ⊗ u_1, a saída <Z_0> colapsa numa
    função só de x_1 (Cap. 2, "Entrelaçamento e termos cruzados"): o segundo
    atributo é codificado, ocupa um
    qubit e é completamente ignorado pela medição.
    """
    wires = list(wires)
    L_var = pnp.shape(weights)[0]
    for camada in range(L_var):
        for i, w in enumerate(wires):
            qml.Rot(
                weights[camada, i, 0],
                weights[camada, i, 1],
                weights[camada, i, 2],
                wires=w,
            )


def weights_shape_local(L_var: int, n_qubits: int) -> tuple[int, ...]:
    """Mesma forma do StronglyEntanglingLayers: (L_var, n, 3)."""
    return (L_var, n_qubits, 3)


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
        descricao="Rot por qubit, sem CNOTs — só para a ablação da hipótese H3",
    ),
}


def get_ansatz(name: str) -> Ansatz:
    if name not in ANSATZE:
        raise ValueError(f"ansatz desconhecido: {name!r}. Opções: {sorted(ANSATZE)}")
    return ANSATZE[name]


def init_weights(
    L_var: int,
    n_qubits: int,
    seed: int = 42,
    escala: float = 0.1,
    forma: tuple[int, ...] | None = None,
):
    """Pesos iniciais pequenos e reprodutíveis.

    A escala pequena (N(0, 0,1^2)) é uma decisão empírica: deixa as rotações
    perto da identidade, mas o circuito não fica perto da identidade, porque
    as CNOTs continuam lá. Não equivale à construção de Grant et al. (2019),
    em que blocos inteiros se cancelam, e não garante evitar barren plateaus.

    ``forma`` permite inicializar um tensor de forma arbitrária (o
    re-uploading tem os pesos ditados pela codificação, não por (L_var, n)).
    """
    if forma is None:
        forma = weights_shape(L_var, n_qubits)
    rng = pnp.random.default_rng(seed)
    return pnp.array(rng.normal(0.0, escala, size=forma), requires_grad=True)
