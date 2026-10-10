"""Codificações de dados clássicos em estados quânticos.

Todas as codificações compartilham a mesma interface para que a comparação —
trocar a codificação mantendo o resto fixo — seja uma troca de string.

O contrato é o da dataclass ``Encoding``:

- ``apply(x, wires, params=None)`` escreve as portas no circuito;
- ``n_qubits(n_features)`` diz quantos qubits a codificação exige;
- ``params_shape(n_features)`` dá a forma do tensor entregue em ``params``
  (``None`` quando a codificação não recebe nenhum);
- ``interleaved=True`` significa que ``apply`` monta o circuito INTEIRO
  (dados e camadas treináveis intercalados) e o ansatz não deve ser chamado
  depois — é o caso do data re-uploading (Cap. 2, "Data re-uploading");
- ``apply_dados`` isola o bloco de dados, para medir a profundidade da
  codificação separadamente da profundidade total (Cap. 3, "Métricas").

Registrar uma codificação nova em ``ENCODINGS`` basta para ela aparecer na
linha de comando e poder entrar na grade de experimentos — não há nenhuma
segunda lista para manter em sincronia.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pennylane as qml

from tccqml.ansatz import weights_shape


@dataclass(frozen=True)
class Encoding:
    """Uma estratégia de codificação x -> psi(x)."""

    name: str
    apply: Callable[..., None]
    n_qubits: Callable[[int], int]
    params_shape: Callable[[int], tuple[int, ...] | None] | None = None
    interleaved: bool = False
    descricao: str = ""
    apply_dados: Callable[..., None] | None = None

    def bloco_de_dados(self) -> Callable[..., None]:
        """Só a parte que depende de x — usada para medir profundidade."""
        return self.apply_dados if self.apply_dados is not None else self.apply


# --------------------------------------------------------------------------
# 1. Angle encoding (Cap. 2, "Angle encoding") — o classificador de referência
# --------------------------------------------------------------------------


def angle_encoding(x, wires: Sequence[int], params=None, rotation: str = "Y") -> None:
    """Cada feature vira o ângulo de uma rotação: x_i -> R(x_i).

    Um qubit por feature. Bloco de dados com profundidade 1 e nenhuma porta
    de dois qubits.
    """
    qml.AngleEmbedding(x, wires=wires, rotation=rotation)


# --------------------------------------------------------------------------
# 2. Amplitude encoding (Cap. 2, "Amplitude encoding")
# --------------------------------------------------------------------------


def amplitude_encoding(x, wires: Sequence[int], params=None) -> None:
    """Os d atributos viram as amplitudes de um estado de ceil(log2 d) qubits.

    Com d = 2 isso é UM único qubit — e não dois, como nas demais. A
    consequência aparece em toda tabela: o StronglyEntanglingLayers sobre um
    qubit não gera nenhuma CNOT e o número de parâmetros cai de 12 para 6.
    Não é bug; é a codificação sendo o que ela é, e por isso `n_qubits` e `p`
    são emitidos em toda linha de resultado.

    O vetor é normalizado antes de virar estado, o que descarta a norma de x:
    dois pontos na mesma direção viram o MESMO estado, e x e -x diferem só
    por fase global. Como
    `data.py` normaliza para [0, pi], todos os pontos caem no primeiro
    quadrante e o ângulo entre dois vetores quaisquer fica em [0, pi/2]. O diagnóstico
    (`diagnostico_amplitude`, em `experiments.py`) mede o efeito nas duas
    normalizações.
    """
    qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)


def _qubits_amplitude(n_features: int) -> int:
    return max(1, math.ceil(math.log2(n_features)))


# --------------------------------------------------------------------------
# 3. Data re-uploading (Cap. 2, "Data re-uploading")
# --------------------------------------------------------------------------


def _reuploading_dados(x, wires: Sequence[int], params=None, R: int = 3) -> None:
    """Só os R blocos S(x), sem as camadas treináveis (medida de custo)."""
    for _ in range(R):
        angle_encoding(x, wires=wires)


def make_reuploading(R: int = 3) -> Encoding:
    """Monta a codificação com re-uploading de `R` blocos de dados.

    Cap. 2, "Data re-uploading":

        U(x, theta) = W(theta_R) S(x) ... W(theta_1) S(x)

    Convenção deste trabalho, fixada aqui para não haver ambiguidade:
    `R` blocos de dados S(x) (cada um um angle encoding) intercalados com
    `R` camadas treináveis W(theta_l), cada W sendo UMA camada do
    StronglyEntanglingLayers. A ordem dentro de cada repetição é S(x) e depois
    W(theta_l).

    `R` é INDEPENDENTE do `L_var` do ansatz: como esta codificação é
    `interleaved`, o ansatz não é aplicado depois, e quem determina a forma dos
    pesos é `params_shape`. Logo o modelo tem 3 * R * n parâmetros.
    """

    def aplicar(x, wires: Sequence[int], params=None) -> None:
        if params is None:
            raise ValueError("re-uploading exige os pesos treináveis em params=")
        wires = list(wires)
        for camada in range(R):
            angle_encoding(x, wires=wires)
            qml.StronglyEntanglingLayers(params[camada : camada + 1], wires=wires)

    def dados(x, wires: Sequence[int], params=None) -> None:
        _reuploading_dados(x, wires, R=R)

    return Encoding(
        name="reuploading",
        apply=aplicar,
        n_qubits=lambda n_features: n_features,
        params_shape=lambda n_features: weights_shape(R, n_features),
        interleaved=True,
        descricao=(
            f"dados reinseridos em {R} blocos intercalados com camadas "
            "treináveis; suporte Omega = {-R,...,R} por atributo, com R = "
            f"{R}"
        ),
        apply_dados=dados,
    )


# --------------------------------------------------------------------------
# 4. Feature map entrelaçado ZZ (Cap. 2, "Feature maps entrelaçados")
# --------------------------------------------------------------------------


def make_zz(r_zz: int = 2) -> Encoding:
    """Feature map entrelaçado, com `r_zz` repetições (padrão 2).

    Cada repetição é:

        H em todos os qubits
        RZ(2 x_i) em cada qubit i
        para cada par i < j:  CNOT(i, j); RZ(-2 (pi - x_i)(pi - x_j)) em j; CNOT(i, j)

    A Hadamard NÃO é decoração. O operador de fase é diagonal na base
    computacional, então sobre |0...0> ele só multiplica o estado por uma fase
    global: sem as Hadamards o circuito fica literalmente independente dos
    dados e <Z_0> vale 1 para qualquer x. O teste
    `test_zz_nao_e_inerte` existe por causa disso.

    É a única codificação do núcleo com portas de dois qubits no bloco de
    dados: r_zz * d * (d - 1) CNOTs, ou 4 para d = 2 e r_zz = 2. A profundidade é
    O(r_zz d^2), porque os pares crescem com o quadrado da dimensão.
    """

    def aplicar(x, wires: Sequence[int], params=None) -> None:
        wires = list(wires)
        for _ in range(r_zz):
            for w in wires:
                qml.Hadamard(wires=w)
            for i, w in enumerate(wires):
                qml.RZ(2 * x[..., i], wires=w)
            for i in range(len(wires)):
                for j in range(i + 1, len(wires)):
                    qml.CNOT(wires=[wires[i], wires[j]])
                    qml.RZ(
                        -2 * (np.pi - x[..., i]) * (np.pi - x[..., j]),
                        wires=wires[j],
                    )
                    qml.CNOT(wires=[wires[i], wires[j]])

    return Encoding(
        name="zz",
        apply=aplicar,
        n_qubits=lambda n_features: n_features,
        descricao=(
            f"feature map entrelaçado com r_ZZ = {r_zz} repetições; "
            "único dos implementados com CNOTs no bloco de dados (d = 2)"
        ),
    )


# --------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------

R_PADRAO = 3
R_ZZ_PADRAO = 2

ENCODINGS = {
    "angle": Encoding(
        name="angle",
        apply=angle_encoding,
        n_qubits=lambda n_features: n_features,
        descricao="um RY por atributo; bloco de profundidade 1, sem CNOTs; Omega = {-1, 0, 1}",
    ),
    "amplitude": Encoding(
        name="amplitude",
        apply=amplitude_encoding,
        n_qubits=_qubits_amplitude,
        descricao="atributos nas amplitudes de ceil(log2 d) qubits; descarta a norma",
    ),
    "reuploading": make_reuploading(R_PADRAO),
    "zz": make_zz(R_ZZ_PADRAO),
}

# Codificações que aceitam parâmetros próprios em get_encoding(nome, **kwargs).
_FABRICAS: dict[str, Callable[..., Encoding]] = {
    "reuploading": make_reuploading,
    "zz": make_zz,
}


def get_encoding(name: str, **kwargs) -> Encoding:
    """Devolve a codificação registrada, opcionalmente reparametrizada.

    >>> get_encoding("reuploading", R=5).params_shape(2)
    (5, 2, 3)
    """
    if name not in ENCODINGS:
        raise ValueError(f"codificação desconhecida: {name!r}. Opções: {sorted(ENCODINGS)}")
    if not kwargs:
        return ENCODINGS[name]
    fabrica = _FABRICAS.get(name)
    if fabrica is None:
        raise ValueError(
            f"codificação {name!r} não aceita parâmetros próprios; recebido: {sorted(kwargs)}"
        )
    return fabrica(**kwargs)
