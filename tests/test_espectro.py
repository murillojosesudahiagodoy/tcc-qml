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
    espectro_2d,
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


def test_tabela_5_e_um_limite_superior():
    """Nenhuma energia fora de Omega^d — é o que a Tabela 5 garante.

    E só isso: com o ansatz congelado, x_1 não chega ao omega previsto no
    re-uploading (ver `test_assimetria_entre_atributos_do_reuploading`), e o
    número de termos medido fica abaixo de (2L + 1)^d.
    """
    tabela = tabela_espectro(L_reups=(1, 2, 3))

    assert tabela["dentro_do_limite"].all(), tabela.to_string(index=False)
    assert (tabela["n_termos_medido"] <= tabela["n_termos_previsto"]).all()

    reup = tabela[tabela["encoding"] == "reuploading"]
    assert list(reup["omega_max_x1"]) == [0, 1, 2]
    assert list(reup["omega_max_x2"]) == [1, 2, 3]
    assert not reup["atinge_limite"].any()


def test_espectro_2d_concorda_com_as_varreduras_1d():
    """As duas medições têm que dar o mesmo omega máximo por atributo."""
    for kwargs in (None, {"L_reup": 2}, {"L_reup": 3}):
        enc = "angle" if kwargs is None else "reuploading"
        clf = model.build(enc, n_features=2, n_layers=2, enc_kwargs=kwargs)
        w, alpha = _pesos_excitados(clf)

        e2 = espectro_2d(clf, w, alpha)
        for i, esp in enumerate(espectro_por_atributo(clf, w, alpha)):
            assert e2.omega_max(i) == esp.omega_max


def test_angle_tem_termos_cruzados_com_ansatz_entrelacado():
    """O termo cos(x_1) cos(x_2) do XOR é cruzado (Seção 2.5.6.3).

    Com CNOTs ele recebe peso; sem CNOTs (ansatz `local`) a saída só depende
    de x_1 e nenhum termo cruzado sobrevive (Eq. 2.83).
    """
    com = model.build("angle", n_features=2, n_layers=2)
    sem = model.build("angle", n_features=2, n_layers=2, ansatz="local")

    assert espectro_2d(com, *_pesos_excitados(com)).n_cruzados > 0
    e_sem = espectro_2d(sem, *_pesos_excitados(sem))
    assert e_sem.n_cruzados == 0
    assert e_sem.omega_max(1) == 0


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


@pytest.mark.parametrize(
    ("encoding", "kwargs", "n_termos"),
    [("angle", None, 6), ("reuploading", {"L_reup": 1}, 2), ("reuploading", {"L_reup": 2}, 10),
     ("reuploading", {"L_reup": 3}, 35)],
)
def test_contagem_de_termos_e_da_arquitetura_e_nao_dos_pesos(encoding, kwargs, n_termos):
    """O número de termos da tab_espectro não é sorte de um jogo de pesos.

    São 40 sorteios: 20 sementes da inicialização de treino (escalada) e 20 de
    pesos uniformes em [0, 2pi). A contagem não muda — é o que o relatório afirma.
    """
    clf = model.build(encoding, n_features=2, n_layers=2, enc_kwargs=kwargs)
    contagens = set()
    for seed in range(42, 62):
        w, alpha, _ = pesos_iniciais(clf, seed=seed)
        uniforme = np.random.default_rng(seed).uniform(0, 2 * np.pi, np.shape(w))
        for pesos in (w * 3.0, uniforme):
            contagens.add(espectro_2d(clf, pesos, alpha).n_termos)

    assert contagens == {n_termos}
