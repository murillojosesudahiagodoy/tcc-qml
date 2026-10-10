"""Testes das contagens de recurso — confronto direto com o texto.

Estes testes não verificam que o código roda, e sim que ele reproduz os
números do TCC (tab:codificacoes, a regra de deslocamento de parâmetro e a
contagem (2p + 1)|B|S por passo). Se um deles falhar, ou o código está errado
ou o texto está.
"""

import numpy as np
import pytest

from tccqml import model
from tccqml.circuit_stats import (
    aval_derivadas_amostra,
    aval_gradiente_amostra,
    n_aval_passo,
    n_shots_passo,
    stats,
    stats_ansatz,
)

X_EXEMPLO = [0.3, 1.2]


def _stats(encoding: str, **kwargs) -> dict:
    clf = model.build(encoding, n_features=2, L_var=2, **kwargs)
    return stats(clf, X_EXEMPLO)


# --------------------------------------------------------------------------
# Recursos do ansatz de referência, sem codificação
# --------------------------------------------------------------------------


def test_ansatz_configuracao_de_referencia():
    """n = 2, L_var = 2: p = 3Ln = 12, 3Ln portas de 1q, Ln de 2q, 6Ln avaliações."""
    s = stats_ansatz(n_qubits=2, L_var=2)

    assert s["n_params_circuito"] == 12
    assert s["gates_1q"] == 12
    assert s["gates_2q"] == 4
    assert s["aval_derivadas_amostra"] == 24


@pytest.mark.parametrize("n,L_var", [(2, 2), (3, 2), (2, 4), (5, 3)])
def test_ansatz_formulas_gerais(n, L_var):
    """As fórmulas 3Ln / 3Ln / Ln / 6Ln valem para qualquer (n, L_var)."""
    s = stats_ansatz(n_qubits=n, L_var=L_var)

    assert s["n_params_circuito"] == 3 * L_var * n
    assert s["gates_1q"] == 3 * L_var * n
    assert s["gates_2q"] == L_var * n
    assert s["aval_derivadas_amostra"] == 6 * L_var * n
    assert s["aval_gradiente_amostra"] == 6 * L_var * n + 1


def test_ansatz_local_nao_tem_portas_de_dois_qubits():
    """A ablação só é válida se ela de fato remover o entrelaçamento."""
    s = stats_ansatz(n_qubits=2, L_var=2, ansatz="local")

    assert s["gates_2q"] == 0
    assert s["n_params_circuito"] == 12  # mesmo p: só o entrelaçamento muda


# --------------------------------------------------------------------------
# Custo do gradiente em hardware: (2p + 1) por amostra, |B| amostras, S shots
# --------------------------------------------------------------------------


def test_eq_2_47_duas_avaliacoes_deslocadas_por_parametro():
    """As 2p avaliações deslocadas dão só as derivadas parciais de f(x)."""
    assert aval_derivadas_amostra(12) == 24


def test_gradiente_do_custo_quadratico_precisa_do_residuo():
    """O gradiente de (f + b - y)^2 pede f(x) sem deslocamento: 2p + 1."""
    assert aval_gradiente_amostra(12) == 25
    assert aval_gradiente_amostra(12) == aval_derivadas_amostra(12) + 1


def test_avaliacoes_e_shots_por_passo_separados():
    """(2p + 1)|B| avaliações e (2p + 1)|B|S shots, em funções distintas."""
    assert n_aval_passo(n_params_circuito=12, batch_size=20) == 500
    assert n_shots_passo(n_params_circuito=12, batch_size=20, shots=1000) == 500_000


def test_o_vies_nao_entra_no_custo_do_gradiente():
    """São 12 parâmetros no circuito, não 13: a derivada do viés é clássica."""
    clf = model.build("angle", n_features=2, L_var=2)
    s = stats(clf, X_EXEMPLO, batch_size=20, shots=1000)

    assert s["n_params_modelo"] == 13
    assert s["n_params_circuito"] == 12
    assert s["aval_derivadas_amostra"] == 24
    assert s["aval_gradiente_amostra"] == 25
    assert s["n_aval_passo"] == 500
    assert s["n_shots_passo"] == 500_000


def test_custo_por_passo_do_angle_com_o_protocolo_padrao():
    """|B| e S saem do protocolo, não de números escolhidos à mão.

    Com o split 180/60/60 são 180 amostras de treino divididas em 9 lotes por
    `np.array_split`, então |B| = 20; com S = 1000 e p = 12 (angle) são
    (2 * 12 + 1) * 20 = 500 avaliações e 500 000 shots por passo.
    """
    from tccqml.config import PADRAO

    assert PADRAO.batch_efetivo == 20
    assert PADRAO.shots == 1000
    clf = model.build("angle", n_features=2, L_var=PADRAO.L_var)
    s = stats(clf, X_EXEMPLO, batch_size=PADRAO.batch_efetivo, shots=PADRAO.shots)

    assert s["n_params_circuito"] == 12
    assert s["n_aval_passo"] == 500
    assert s["n_shots_passo"] == 500_000


# --------------------------------------------------------------------------
# Cap. 3, "Métricas" — bloco de codificação e circuito completo, separados
# --------------------------------------------------------------------------


def test_contagens_do_angle_bloco_e_circuito_completo():
    """Angle com L_var = 2 e d = 2: o que o Cap. 3 ("Métricas") manda reportar.

    O bloco de codificação são dois RY em paralelo (2 portas, profundidade 1,
    nenhuma CNOT); as 4 CNOTs do circuito completo são todas do ansatz.
    """
    clf = model.build("angle", n_features=2, L_var=2)
    s = stats(clf, X_EXEMPLO)

    assert s["gates_1q_encoding"] == 2
    assert s["gates_2q_encoding"] == 0
    assert s["gates_total_encoding"] == 2
    assert s["depth_encoding"] == 1
    assert s["n_qubits"] == 2
    assert s["gates_2q"] == 4


@pytest.mark.parametrize("encoding", ["angle", "amplitude", "reuploading", "zz"])
def test_portas_totais_somam_1q_e_2q(encoding):
    """Total = 1q + 2q, no circuito completo e no bloco de codificação."""
    kwargs = {"enc_kwargs": {"R": 3}} if encoding == "reuploading" else {}
    s = _stats(encoding, **kwargs)

    assert s["gates_1q"] + s["gates_2q"] == s["gates_total"]
    assert s["gates_1q_encoding"] + s["gates_2q_encoding"] == s["gates_total_encoding"]


def test_colunas_de_custo_cobrem_a_secao_3_8():
    """O CSV da grade grava o que o Cap. 3 ("Métricas") promete, para os dois níveis."""
    from tccqml.experiments import COLUNAS_CUSTO

    for coluna in ("n_qubits", "depth", "gates_total", "gates_2q",
                   "depth_encoding", "gates_total_encoding", "gates_2q_encoding"):
        assert coluna in COLUNAS_CUSTO


# --------------------------------------------------------------------------
# tab:codificacoes — recursos por codificação, com d = 2
# --------------------------------------------------------------------------


def test_tab_codificacoes_angle():
    """1 qubit por atributo, profundidade 1, nenhuma porta de dois qubits."""
    s = _stats("angle")

    assert s["n_qubits"] == 2
    assert s["depth_encoding"] == 1
    assert s["gates_2q_encoding"] == 0
    assert s["n_params_circuito"] == 12


def test_tab_codificacoes_amplitude():
    """ceil(log2 2) = 1 qubit, e por isso p cai de 12 para 6.

    A profundidade medida da codificação é 2, e não o "uma R_y" da leitura
    teórica: a rotina genérica de preparação de estado do PennyLane emite
    também uma RZ para a fase relativa, que aqui é identicamente nula porque as
    amplitudes são reais e não negativas. Contar o que o circuito de fato tem é
    a decisão do Cap. 2 ("Modelo de recursos"); o O(d) da tab:codificacoes é assintótico.
    """
    s = _stats("amplitude")

    assert s["n_qubits"] == 1
    assert s["gates_2q"] == 0  # um qubit não admite porta de dois qubits
    assert s["n_params_circuito"] == 6
    assert s["depth_encoding"] == 2


def test_tab_codificacoes_reuploading():
    """Profundidade do bloco de dados = R, e nenhuma CNOT nele."""
    for R in (1, 2, 3, 5):
        s = _stats("reuploading", enc_kwargs={"R": R})

        assert s["n_qubits"] == 2
        assert s["depth_encoding"] == R
        assert s["gates_2q_encoding"] == 0
        assert s["n_params_circuito"] == 3 * R * 2


def test_tab_codificacoes_zz():
    """r d (d-1) = 4 CNOTs no bloco de dados — o único do núcleo que tem."""
    s = _stats("zz")

    assert s["n_qubits"] == 2
    assert s["gates_2q_encoding"] == 4
    assert s["n_params_circuito"] == 12


def test_zz_tem_profundidade_quadratica_na_dimensao():
    """O(r d^2), e não O(r d): os pares crescem com o quadrado da dimensão."""
    profundidades = []
    for d in (2, 3, 4):
        clf = model.build("zz", n_features=d, L_var=2)
        profundidades.append(stats(clf, np.full(d, 0.5))["depth_encoding"])

    # Crescimento claramente super-linear entre d = 2 e d = 4.
    assert profundidades[2] > 2 * profundidades[0]


# --------------------------------------------------------------------------
# Decomposição
# --------------------------------------------------------------------------


def test_contagem_usa_o_circuito_decomposto():
    """Sem decompor, o Rot contaria como UMA porta e daria 4 em vez de 12.

    É a exigência do Cap. 2 ("Modelo de recursos"): sem ela, "prepare |psi(x)>" apareceria como
    uma porta só e a comparação de custo perderia o sentido.
    """
    s = stats_ansatz(n_qubits=2, L_var=2)

    assert s["gates_1q"] == 12, "o Rot precisa virar RZ-RY-RZ antes de contar"
    assert "Rot" not in s["gate_types"]
