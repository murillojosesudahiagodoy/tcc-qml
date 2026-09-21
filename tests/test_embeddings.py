"""Testes das codificações (Etapa 4 / Etapa 9)."""

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from tccqml.embeddings import ENCODINGS, get_encoding

NUCLEO = ["angle", "amplitude", "reuploading", "zz"]


def _estado(encoding: str, x, n_qubits: int, params=None):
    enc = get_encoding(encoding)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuito(x, params=None):
        enc.apply(x, wires=range(n_qubits), params=params)
        return qml.state()

    return circuito(pnp.array(x, requires_grad=False), params)


def _expval(encoding: str, x, n_qubits: int, params=None):
    enc = get_encoding(encoding)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuito(x, params=None):
        enc.apply(x, wires=range(n_qubits), params=params)
        return qml.expval(qml.PauliZ(0))

    return float(circuito(pnp.array(x, requires_grad=False), params))


# --------------------------------------------------------------------------
# Interface comum
# --------------------------------------------------------------------------


def test_angle_usa_um_qubit_por_feature():
    enc = get_encoding("angle")
    assert enc.n_qubits(2) == 2
    assert enc.n_qubits(4) == 4


def test_amplitude_usa_log2_qubits():
    """ceil(log2 d): 1 qubit para 2 atributos, 2 para 4 (Seção 2.4.4)."""
    enc = get_encoding("amplitude")
    assert enc.n_qubits(2) == 1
    assert enc.n_qubits(4) == 2
    assert enc.n_qubits(5) == 3


def test_codificacao_desconhecida_falha():
    with pytest.raises(ValueError, match="desconhecida"):
        get_encoding("teleporte")


def test_codificacao_sem_parametros_recusa_kwargs():
    with pytest.raises(ValueError, match="não aceita parâmetros"):
        get_encoding("angle", L_reup=3)


def test_registro_tem_interface_uniforme():
    """A Etapa 9 depende disso: toda codificação com a mesma assinatura."""
    for nome, enc in ENCODINGS.items():
        assert enc.name == nome
        assert callable(enc.apply)
        assert callable(enc.n_qubits)
        assert enc.descricao, f"{nome} precisa de descrição: a CLI lê daqui"


def test_registro_cobre_as_quatro_da_etapa_9():
    assert set(NUCLEO) <= set(ENCODINGS)


# --------------------------------------------------------------------------
# Propriedades que toda codificação precisa ter
# --------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", NUCLEO)
def test_produz_estado_normalizado(encoding):
    """Norma 1 é a condição de estado físico. Se quebrar, tudo depois é lixo."""
    enc = get_encoding(encoding)
    n_qubits = enc.n_qubits(2)
    params = (
        pnp.zeros(enc.params_shape(2), requires_grad=True) + 0.3
        if enc.params_shape is not None
        else None
    )

    estado = _estado(encoding, [0.3, 1.2], n_qubits, params)

    assert np.isclose(np.sum(np.abs(estado) ** 2), 1.0)


@pytest.mark.parametrize("encoding", NUCLEO)
def test_entradas_distintas_geram_saidas_distintas(encoding):
    """Se dois x diferentes colidem, a codificação perdeu informação.

    No caso do ZZ isto é mais que higiene: o operador de fase é diagonal e,
    sem as Hadamards, o circuito fica literalmente independente dos dados sem
    lançar erro nenhum (Seção 2.4.6).
    """
    enc = get_encoding(encoding)
    n_qubits = enc.n_qubits(2)
    params = (
        pnp.zeros(enc.params_shape(2), requires_grad=True) + 0.3
        if enc.params_shape is not None
        else None
    )

    a = _expval(encoding, [0.5, 1.0], n_qubits, params)
    b = _expval(encoding, [2.0, 0.4], n_qubits, params)

    assert abs(a - b) > 1e-6


# --------------------------------------------------------------------------
# Angle
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Amplitude
# --------------------------------------------------------------------------


def test_amplitude_descarta_a_norma():
    """k(x, cx) = 1 para todo c > 0: só a DIREÇÃO sobrevive (Seção 2.4.4.1).

    É a razão pela qual a normalização dos dados interage com esta codificação
    e precisa do diagnóstico de normalização — dois pontos distantes em norma viram
    exatamente o mesmo estado.
    """
    a = _estado("amplitude", [0.3, 1.2], 1)
    b = _estado("amplitude", [1.5, 6.0], 1)  # o mesmo vetor, 5x maior

    assert np.isclose(abs(np.vdot(a, b)) ** 2, 1.0)


# --------------------------------------------------------------------------
# Re-uploading
# --------------------------------------------------------------------------


def test_reuploading_e_interleaved_e_dita_a_forma_dos_pesos():
    enc = get_encoding("reuploading", L_reup=4)

    assert enc.interleaved
    assert tuple(enc.params_shape(2)) == (4, 2, 3)


def test_reuploading_sem_pesos_falha_alto():
    """Silenciosamente aplicar só os dados seria um bug difícil de achar."""
    enc = get_encoding("reuploading")
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuito(x):
        enc.apply(x, wires=range(2), params=None)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError, match="params"):
        circuito(pnp.array([0.3, 1.2], requires_grad=False))


# --------------------------------------------------------------------------
# Feature map ZZ
# --------------------------------------------------------------------------


def test_zz_sem_hadamard_seria_inerte():
    """A armadilha da Seção 2.4.6, demonstrada.

    O operador de fase do ZZ é diagonal na base computacional: sobre |0...0>
    ele só produz uma fase global. Este teste constrói o feature map SEM as
    Hadamards e confirma que <Z_0> = 1 para qualquer entrada — o modelo ficaria
    independente dos dados sem dar nenhum sinal de erro. O teste irmão
    `test_entradas_distintas_geram_saidas_distintas[zz]` garante que a versão
    de verdade não cai nisso.
    """
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def sem_hadamard(x):
        for i in range(2):
            qml.RZ(2 * x[i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.RZ(-2 * (np.pi - x[0]) * (np.pi - x[1]), wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    a = float(sem_hadamard(pnp.array([0.5, 1.0], requires_grad=False)))
    b = float(sem_hadamard(pnp.array([2.0, 0.4], requires_grad=False)))

    assert np.isclose(a, 1.0) and np.isclose(b, 1.0)


def test_zz_tem_o_numero_de_cnots_previsto():
    """r * d * (d - 1) = 4 CNOTs para d = 2 e r = 2 (Eq. 2.66)."""
    enc = get_encoding("zz", r=2)
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuito(x):
        enc.apply(x, wires=range(2))
        return qml.expval(qml.PauliZ(0))

    specs = qml.specs(circuito)(pnp.array([0.3, 1.2], requires_grad=False))
    assert specs["resources"].gate_types.get("CNOT", 0) == 4
