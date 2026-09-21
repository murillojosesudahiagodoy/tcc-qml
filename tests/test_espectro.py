"""Testes do espectro de Fourier — o resultado teórico central.

A Tabela 5 (p. 52) afirma que a CODIFICAÇÃO determina quais frequências o
modelo consegue representar. Estes testes medem por FFT e confrontam.
"""

import numpy as np
import pytest

from tccqml import model
from tccqml.espectro import (
    ENCODINGS_COM_ESPECTRO,
    espectro,
    espectro_por_atributo,
    n_termos_previsto,
    omega_previsto,
    tabela_espectro,
)
from tccqml.model import pesos_iniciais


def _pesos_excitados(clf, escala: float = 3.0):
    """Pesos grandes o bastante para excitar todo o espectro acessível.

    A inicialização de treino é pequena de propósito (barren plateau), mas com
    coeficientes minúsculos uma frequência acessível pode ficar abaixo do
    limiar e parecer ausente. O que se mede aqui é o CONJUNTO de frequências,
    não um jogo de pesos específico.
    """
    w, alpha, _ = pesos_iniciais(clf)
    return w * escala, alpha


def test_angle_nao_tem_energia_acima_da_frequencia_1():
    """Omega = {-1, 0, 1}: o piso contra o qual o re-uploading é medido.

    Se este teste falhar, a medição está errada — não a teoria.
    """
    clf = model.build("angle", n_features=2, n_layers=2)
    w, alpha = _pesos_excitados(clf)

    for esp in espectro_por_atributo(clf, w, alpha):
        assert esp.omega_max <= 1
        assert np.all(esp.amplitudes[2:] < 1e-6)


@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_reuploading_alcanca_a_frequencia_L_e_nada_acima(L):
    """Omega = {-L, ..., L}: energia até L, praticamente zero acima."""
    clf = model.build("reuploading", n_features=2, n_layers=2, enc_kwargs={"L_reup": L})
    w, alpha = _pesos_excitados(clf)

    espectros = espectro_por_atributo(clf, w, alpha)
    alcancado = max(e.omega_max for e in espectros)

    assert alcancado == L
    for esp in espectros:
        assert np.all(esp.amplitudes[L + 1 :] < 1e-6), "energia acima de L não deveria existir"


def test_espectro_cresce_com_L():
    """Mais blocos de dados, mais frequências — é a razão de ser da codificação."""
    alcancados = []
    for L in (1, 2, 3):
        clf = model.build("reuploading", n_features=2, n_layers=2, enc_kwargs={"L_reup": L})
        w, alpha = _pesos_excitados(clf)
        alcancados.append(max(e.omega_max for e in espectro_por_atributo(clf, w, alpha)))

    assert alcancados == [1, 2, 3]


def test_assimetria_entre_atributos_do_reuploading():
    """Uma consequência MEDIDA do ansatz congelado, que vale reportar.

    Com dois qubits, o par de CNOTs em anel leva Z_0 exatamente em Z_1, então o
    último bloco S(x) não alcança o atributo 0: varrendo x_0 o espectro vai só
    até L - 1. O limite teórico é sobre o que a codificação torna acessível; o
    que se perde num atributo é do ansatz.
    """
    L = 3
    clf = model.build("reuploading", n_features=2, n_layers=2, enc_kwargs={"L_reup": L})
    w, alpha = _pesos_excitados(clf)

    esp_x0, esp_x1 = espectro_por_atributo(clf, w, alpha)

    assert esp_x1.omega_max == L
    assert esp_x0.omega_max == L - 1


def test_tabela_5_e_reproduzida_por_completo():
    """A linha a linha da Tabela 5 para as codificações em que ela se aplica."""
    tabela = tabela_espectro(L_reups=(1, 2, 3))

    assert tabela["confere"].all(), tabela.to_string(index=False)


def test_49_termos_contra_p_parametros():
    """O número do Capítulo 4 (Seção 2.5.6.1): d = 2, L = 3 dá (2L+1)^2 = 49.

    São 49 coeficientes de Fourier contra 18 parâmetros treináveis. Eles não
    podem ser escolhidos de forma independente — é a explicação teórica da
    saturação da Previsão 5.
    """
    assert n_termos_previsto("reuploading", d=2, L_reup=3) == 49
    assert n_termos_previsto("angle", d=2) == 9

    clf = model.build("reuploading", n_features=2, n_layers=2, enc_kwargs={"L_reup": 3})
    assert clf.n_params_circuito == 18


def test_omega_previsto_declara_onde_nao_se_aplica():
    """amplitude e zz não têm série de Fourier nos atributos — e isso é explícito."""
    assert omega_previsto("angle") == {-1, 0, 1}
    assert omega_previsto("reuploading", L_reup=2) == {-2, -1, 0, 1, 2}
    assert omega_previsto("amplitude") is None
    assert omega_previsto("zz") is None
    assert set(ENCODINGS_COM_ESPECTRO) == {"angle", "reuploading"}


def test_espectro_e_reprodutivel():
    clf = model.build("angle", n_features=2, n_layers=2)
    w, alpha = _pesos_excitados(clf)

    a = espectro(clf, w, alpha)
    b = espectro(clf, w, alpha)

    assert np.allclose(a.amplitudes, b.amplitudes)
