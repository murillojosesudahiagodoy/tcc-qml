"""Codificações de dados clássicos em estados quânticos (Etapa 4).

Todas as codificações compartilham a mesma interface para que a Etapa 9 —
trocar a codificação mantendo o resto fixo — seja uma troca de string.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import pennylane as qml


@dataclass(frozen=True)
class Encoding:
    """Uma estratégia de codificação x -> psi(x)."""

    name: str
    apply: Callable[..., None]
    n_qubits: Callable[[int], int]


def angle_encoding(x, wires: Sequence[int], rotation: str = "Y") -> None:
    """Cada feature vira o ângulo de uma rotação: x_i -> R(x_i).

    Um qubit por feature. Circuito raso: profundidade 1, nenhuma porta
    de dois qubits. É o piso de custo contra o qual as outras comparam.
    """
    qml.AngleEmbedding(x, wires=wires, rotation=rotation)


ENCODINGS = {
    "angle": Encoding(
        name="angle",
        apply=angle_encoding,
        n_qubits=lambda n_features: n_features,
    ),
}


def get_encoding(name: str) -> Encoding:
    if name not in ENCODINGS:
        raise ValueError(f"codificação desconhecida: {name!r}. Opções: {sorted(ENCODINGS)}")
    return ENCODINGS[name]