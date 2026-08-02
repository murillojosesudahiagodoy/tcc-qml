"""Testes das codificações (Etapa 4)."""

import pennylane as qml
import pytest
from pennylane import numpy as pnp

from tccqml.embeddings import ENCODINGS, get_encoding


def test_angle_usa_um_qubit_por_feature():
    enc = get_encoding("angle")
    assert enc.n_qubits(2) == 2
    assert enc.n_qubits(4) == 4


def test_angle_produz_estado_normalizado():
    """Norma 1 é a condição de estado físico. Se quebrar, tudo depois é lixo."""
    enc = get_encoding("angle")
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuito(x):
        enc.apply(x, wires=range(2))
        return qml.state()

    estado = circuito(pnp.array([0.3, 1.2], requires_grad=False))
    assert pnp.isclose(pnp.sum(pnp.abs(estado) ** 2), 1.0)


def test_angle_e_raso_e_sem_portas_de_dois_qubits():
    """O piso de custo contra o qual as outras codificações são comparadas."""
    enc = get_encoding("angle")
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuito(x):
        enc.apply(x, wires=range(2))
        return qml.expval(qml.PauliZ(0))

    specs = qml.specs(circuito)(pnp.array([0.3, 1.2], requires_grad=False))
    contagem = specs["resources"].gate_types
    assert sum(v for k, v in contagem.items() if k in {"CNOT", "CZ"}) == 0


def test_entradas_distintas_geram_estados_distintos():
    """Injetividade em [0, pi]: se dois x diferentes colidem, perde-se informação."""
    enc = get_encoding("angle")
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuito(x):
        enc.apply(x, wires=[0])
        return qml.expval(qml.PauliZ(0))

    a = float(circuito(pnp.array([0.5], requires_grad=False)))
    b = float(circuito(pnp.array([2.0], requires_grad=False)))
    assert abs(a - b) > 1e-6


def test_codificacao_desconhecida_falha():
    with pytest.raises(ValueError, match="desconhecida"):
        get_encoding("teleporte")


def test_registro_tem_interface_uniforme():
    """A Etapa 9 depende disso: toda codificação com a mesma assinatura."""
    for nome, enc in ENCODINGS.items():
        assert enc.name == nome
        assert callable(enc.apply)
        assert callable(enc.n_qubits)