"""Testes das verificações sobre os dados (controle linear e limiar de g).

Protegem três coisas: as nove funções são as das fórmulas, o limiar é
escolhido só no treino, e os CSVs cobrem todas as combinações com as mesmas
partições da grade principal.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from tccqml.config import PADRAO
from tccqml.data import load_dataset
from tccqml.tabelas import tab_verificacoes
from tccqml.verificacoes import (
    NOMES_CARACTERISTICAS,
    aplicar_limiar,
    caracteristicas_fourier,
    escolher_limiar,
    g,
    limiar_circles_um,
    rodar_controle_linear,
    rodar_limiar_circles,
)

PI = np.pi


# --------------------------------------------------------------------------
# (a) as nove características batem com as fórmulas
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x, esperado",
    [
        #            1  c1 s1 c2 s2 c1c2 c1s2 s1c2 s1s2
        ((PI / 2, PI / 2), [1, 0, 1, 0, 1, 0, 0, 0, 1]),
        ((0.0, PI / 2), [1, 1, 0, 0, 1, 0, 1, 0, 0]),
        ((PI, 0.0), [1, -1, 0, 1, 0, -1, 0, 0, 0]),
        ((0.0, 0.0), [1, 1, 0, 1, 0, 1, 0, 0, 0]),
    ],
)
def test_caracteristicas_em_pontos_conhecidos(x, esperado):
    obtido = caracteristicas_fourier(np.array([x]))[0]

    np.testing.assert_allclose(obtido, esperado, atol=1e-12)


def test_caracteristicas_seguem_a_ordem_dos_nomes():
    """Ponto genérico: cada coluna é a fórmula que o nome diz."""
    x1, x2 = 0.3, 1.2
    formulas = {
        "1": 1.0,
        "cos x1": np.cos(x1),
        "sin x1": np.sin(x1),
        "cos x2": np.cos(x2),
        "sin x2": np.sin(x2),
        "cos x1 cos x2": np.cos(x1) * np.cos(x2),
        "cos x1 sin x2": np.cos(x1) * np.sin(x2),
        "sin x1 cos x2": np.sin(x1) * np.cos(x2),
        "sin x1 sin x2": np.sin(x1) * np.sin(x2),
    }
    obtido = caracteristicas_fourier(np.array([[x1, x2]]))[0]

    assert len(NOMES_CARACTERISTICAS) == 9
    np.testing.assert_allclose(obtido, [formulas[n] for n in NOMES_CARACTERISTICAS])


def test_g_no_centro_e_nos_cantos():
    """g = sin x1 + sin x2, como no texto: vai de 0 nos cantos a 2 no centro."""
    X = np.array([[PI / 2, PI / 2], [0.0, 0.0], [0.0, PI / 2]])

    np.testing.assert_allclose(g(X), [2.0, 0.0, 1.0], atol=1e-12)


# --------------------------------------------------------------------------
# (b) o limiar é escolhido só com o treino
# --------------------------------------------------------------------------


def test_limiar_e_escolhido_no_treino_nao_no_teste():
    """No treino a fronteira está em 0.5; no teste, em 0.8. Tem que sair ~0.5."""
    g_treino = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.9, 0.95])
    y_treino = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    g_teste = np.array([0.1, 0.6, 0.7, 0.75, 0.85, 0.9])
    y_teste = np.array([0, 0, 0, 0, 1, 1])

    limiar, sentido = escolher_limiar(g_treino, y_treino)

    assert sentido == ">"
    assert 0.4 < limiar < 0.6
    # O limiar ótimo do teste daria 100% nele; o do treino não, e é ele que vale.
    assert (aplicar_limiar(g_teste, limiar, sentido) == y_teste).mean() < 1.0
    assert (aplicar_limiar(g_teste, 0.8, ">") == y_teste).mean() == 1.0


def test_escolher_limiar_nao_aceita_outros_conjuntos():
    """A assinatura só tem (g_treino, y_treino): validação e teste não entram."""
    import inspect

    assert list(inspect.signature(escolher_limiar).parameters) == ["g_treino", "y_treino"]


def test_limiar_detecta_o_sentido_invertido():
    g = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([1, 1, 1, 0, 0, 0])

    limiar, sentido = escolher_limiar(g, y)

    assert sentido == "<"
    assert (aplicar_limiar(g, limiar, sentido) == y).all()


def test_empate_fica_no_meio_do_intervalo():
    """Com um vão largo entre as classes, o limiar não encosta em nenhuma."""
    g = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])

    limiar, _ = escolher_limiar(g, y)

    assert limiar == pytest.approx(0.5)


def test_limiar_circles_usa_so_o_treino(monkeypatch):
    """Embaralhar os rótulos de validação e teste não muda o limiar."""
    from tccqml import verificacoes

    original = verificacoes._carregar

    def com_avaliacao_embaralhada(*args, **kwargs):
        ds = original(*args, **kwargs)
        rng = np.random.default_rng(0)
        ds.y_val = rng.permutation(ds.y_val)
        ds.y_test = 1 - ds.y_test
        return ds

    normal = limiar_circles_um(42, PADRAO.noise)
    monkeypatch.setattr(verificacoes, "_carregar", com_avaliacao_embaralhada)
    embaralhado = limiar_circles_um(42, PADRAO.noise)

    assert embaralhado["limiar"] == normal["limiar"]
    assert embaralhado["sentido"] == normal["sentido"]
    assert embaralhado["acc_treino"] == normal["acc_treino"]
    assert embaralhado["acc_teste"] == pytest.approx(1 - normal["acc_teste"])


# --------------------------------------------------------------------------
# (c) CSVs com todas as combinações, nas partições da grade
# --------------------------------------------------------------------------


def test_controle_linear_csv_tem_todas_as_combinacoes(tmp_path):
    rodar_controle_linear(out=tmp_path, verbose=False)

    csv = pd.read_csv(tmp_path / "metrics" / "controle_linear.csv")
    resumo = pd.read_csv(tmp_path / "metrics" / "resumo_controle_linear.csv")
    esperado = set(itertools.product(PADRAO.datasets_grade, PADRAO.sementes))

    assert list(csv.columns) == ["dataset", "seed", "acc_treino", "acc_val", "acc_teste"]
    assert len(csv) == len(esperado)
    assert set(csv[["dataset", "seed"]].itertuples(index=False, name=None)) == esperado
    assert set(resumo["dataset"]) == set(PADRAO.datasets_grade)
    assert {"acc_teste_mean", "acc_teste_std"} <= set(resumo.columns)


def test_limiar_csv_tem_todas_as_combinacoes(tmp_path):
    rodar_limiar_circles(out=tmp_path, verbose=False)

    csv = pd.read_csv(tmp_path / "metrics" / "limiar_circles.csv")
    esperado = set(itertools.product(("sem_ruido", "com_ruido"), PADRAO.sementes))

    assert len(csv) == len(esperado)
    assert set(csv[["versao", "seed"]].itertuples(index=False, name=None)) == esperado
    for coluna in ("limiar", "sentido", "acc_treino", "acc_val", "acc_teste",
                   "g_media_c0", "g_desvio_c0", "g_media_c1", "g_desvio_c1",
                   "g_min_c0", "g_max_c0", "g_min_c1", "g_max_c1"):
        assert coluna in csv.columns
    assert set(csv.loc[csv["versao"] == "com_ruido", "ruido"]) == {PADRAO.noise}
    assert set(csv.loc[csv["versao"] == "sem_ruido", "ruido"]) == {0.0}
    assert (tmp_path / "metrics" / "resumo_limiar_circles.csv").exists()


def test_mesmas_particoes_da_grade():
    """As verificações carregam os dados exatamente como `load_dataset` padrão."""
    from tccqml.verificacoes import _carregar

    for dataset in PADRAO.datasets_grade:
        a, b = _carregar(dataset, 43, PADRAO), load_dataset(dataset, seed=43)
        np.testing.assert_array_equal(a.X_test, b.X_test)
        np.testing.assert_array_equal(a.y_train, b.y_train)


# --------------------------------------------------------------------------
# tabela e figura
# --------------------------------------------------------------------------


def test_tab_verificacoes():
    controle = pd.DataFrame(
        [{"dataset": d, "seed": s, "acc_teste": 0.8} for d in ("xor", "moons", "circles") for s in (42, 43)]
    )
    limiar = pd.DataFrame(
        [{"versao": v, "ruido": r, "seed": s, "acc_teste": a}
         for v, r, a in (("sem_ruido", 0.0, 1.0), ("com_ruido", 0.15, 0.9)) for s in (42, 43)]
    )

    tabela, tex = tab_verificacoes(controle, limiar)

    assert list(tabela.columns) == ["Verificação", "\\textit{Xor}", "\\textit{Moons}", "\\textit{Circles}"]
    assert len(tabela) == 3
    assert tabela.iloc[1]["\\textit{Circles}"] == "1,000 $\\pm$ 0,000"
    assert tabela.iloc[2]["\\textit{Xor}"] == "--"
    assert "ruído 0,15" in tabela.iloc[2]["Verificação"]
    assert "não prova" in tex and "treino" in tex


def test_figura_do_limiar(tmp_path):
    from tccqml.figuras import gerar_todas

    nomes = {p.name for p in gerar_todas(out=tmp_path, verbose=False)}

    assert "limiar-circles.pdf" in nomes
