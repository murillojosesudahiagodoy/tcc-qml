"""Testes das contagens de recurso — confronto direto com o texto.

Estes são os testes que importam mais no repositório: eles não verificam que o
código roda, e sim que ele REPRODUZ os números impressos no TCC. Se um deles
falhar, ou o código está errado ou o texto está — e vale muito saber qual.
"""

import numpy as np
import pytest

from tccqml import model
from tccqml.circuit_stats import (
    avaliacoes_por_gradiente,
    n_execucoes_hardware,
    stats,
    stats_ansatz,
)

X_EXEMPLO = [0.3, 1.2]


def _stats(encoding: str, **kwargs) -> dict:
    clf = model.build(encoding, n_features=2, n_layers=2, **kwargs)
    return stats(clf, X_EXEMPLO)


# --------------------------------------------------------------------------
# Tabela 2 (p. 31) — recursos do ansatz de referência
# --------------------------------------------------------------------------


def test_tabela_2_configuracao_de_referencia():
    """n = 2, L = 2: p = 3Ln = 12, 3Ln portas de 1q, Ln de 2q, 6Ln avaliações."""
    s = stats_ansatz(n_qubits=2, n_layers=2)

    assert s["n_params_circuito"] == 12
    assert s["gates_1q"] == 12
    assert s["gates_2q"] == 4
    assert s["avaliacoes_por_gradiente"] == 24


@pytest.mark.parametrize("n,L", [(2, 2), (3, 2), (2, 4), (5, 3)])
def test_tabela_2_formulas_gerais(n, L):
    """As fórmulas 3Ln / 3Ln / Ln / 6Ln valem para qualquer (n, L)."""
    s = stats_ansatz(n_qubits=n, n_layers=L)

    assert s["n_params_circuito"] == 3 * L * n
    assert s["gates_1q"] == 3 * L * n
    assert s["gates_2q"] == L * n
    assert s["avaliacoes_por_gradiente"] == 6 * L * n


def test_ansatz_local_nao_tem_portas_de_dois_qubits():
    """A ablação só é válida se ela de fato remover o entrelaçamento."""
    s = stats_ansatz(n_qubits=2, n_layers=2, ansatz="local")

    assert s["gates_2q"] == 0
    assert s["n_params_circuito"] == 12  # mesmo p: só o entrelaçamento muda


# --------------------------------------------------------------------------
# Eq. 2.47 e Eq. 2.48 — custo do gradiente
# --------------------------------------------------------------------------


def test_eq_2_47_duas_avaliacoes_por_parametro():
    assert avaliacoes_por_gradiente(12) == 24


def test_eq_2_48_dá_504_000_na_configuracao_de_referencia():
    """N_circ = 2 p |B| shots = 2 * 12 * 21 * 1000 = 504 000 (p. 36 do texto)."""
    assert n_execucoes_hardware(n_params_circuito=12, batch_size=21, shots=1000) == 504_000


def test_o_vies_nao_entra_no_custo_do_gradiente():
    """São 12 parâmetros no circuito, não 13: a derivada do viés é clássica."""
    clf = model.build("angle", n_features=2, n_layers=2)
    s = stats(clf, X_EXEMPLO, batch_size=21, shots=1000)

    assert s["n_params_modelo"] == 13
    assert s["n_params_circuito"] == 12
    assert s["n_execucoes_hardware"] == 504_000


def test_batch_efetivo_do_protocolo_e_21():
    """O |B| = 21 da Eq. 2.48 sai do protocolo, não de um número escolhido à mão.

    São 210 amostras de treino divididas em 10 lotes por `np.array_split`.
    """
    from tccqml.config import PADRAO

    assert PADRAO.batch_efetivo == 21


# --------------------------------------------------------------------------
# Tabela 3 — recursos por codificação, com d = 2
# --------------------------------------------------------------------------


def test_tabela_3_angle():
    """1 qubit por atributo, profundidade 1, nenhuma porta de dois qubits."""
    s = _stats("angle")

    assert s["n_qubits"] == 2
    assert s["depth_encoding"] == 1
    assert s["gates_2q_encoding"] == 0
    assert s["n_params_circuito"] == 12


def test_tabela_3_amplitude():
    """ceil(log2 2) = 1 qubit, e por isso p cai de 12 para 6.

    A profundidade medida da codificação é 2, e não o "uma R_y" da leitura
    teórica: a rotina genérica de preparação de estado do PennyLane emite
    também uma RZ para a fase relativa, que aqui é identicamente nula porque as
    amplitudes são reais e não negativas. Contar o que o circuito de fato tem é
    a decisão da Seção 2.6.1.1; o O(d) da Tabela 3 é assintótico.
    """
    s = _stats("amplitude")

    assert s["n_qubits"] == 1
    assert s["gates_2q"] == 0  # um qubit não admite porta de dois qubits
    assert s["n_params_circuito"] == 6
    assert s["depth_encoding"] == 2


def test_tabela_3_reuploading():
    """Profundidade do bloco de dados = L, e nenhuma CNOT nele."""
    for L in (1, 2, 3, 5):
        s = _stats("reuploading", enc_kwargs={"L_reup": L})

        assert s["n_qubits"] == 2
        assert s["depth_encoding"] == L
        assert s["gates_2q_encoding"] == 0
        assert s["n_params_circuito"] == 3 * L * 2


def test_tabela_3_zz():
    """r d (d-1) = 4 CNOTs no bloco de dados — o único do núcleo que tem."""
    s = _stats("zz")

    assert s["n_qubits"] == 2
    assert s["gates_2q_encoding"] == 4
    assert s["n_params_circuito"] == 12


def test_zz_tem_profundidade_quadratica_na_dimensao():
    """O(r d^2), e não O(r d): os pares crescem com o quadrado da dimensão."""
    profundidades = []
    for d in (2, 3, 4):
        clf = model.build("zz", n_features=d, n_layers=2)
        profundidades.append(stats(clf, np.full(d, 0.5))["depth_encoding"])

    # Crescimento claramente super-linear entre d = 2 e d = 4.
    assert profundidades[2] > 2 * profundidades[0]


# --------------------------------------------------------------------------
# Decomposição
# --------------------------------------------------------------------------


def test_contagem_usa_o_circuito_decomposto():
    """Sem decompor, o Rot contaria como UMA porta e daria 4 em vez de 12.

    É a exigência da Seção 2.6.1.1: sem ela, "prepare |psi(x)>" apareceria como
    uma porta só e a comparação de custo perderia o sentido.
    """
    s = stats_ansatz(n_qubits=2, n_layers=2)

    assert s["gates_1q"] == 12, "o Rot precisa virar RZ-RY-RZ antes de contar"
    assert "Rot" not in s["gate_types"]
