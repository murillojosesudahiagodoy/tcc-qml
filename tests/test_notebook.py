"""Testes do apoio ao notebook: as listas dele não podem divergir da CLI nem de results/.

Nenhum destes testes precisa de IPython: o módulo só o importa ao exibir algo.
"""

from pathlib import Path

import pytest

from tccqml import notebook as nb
from tccqml.cli import _parser

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_tabelas_do_notebook_sao_as_versionadas():
    assert set(nb.TABELAS) == {p.stem for p in (RESULTS / "tables").glob("*.tex")}


def test_figuras_do_notebook_sao_as_versionadas():
    assert set(nb.FIGURAS) == {p.name for p in (RESULTS / "figures").glob("*.pdf")}


def test_todo_artefato_tem_lugar_no_tcc():
    assert set(nb.ONDE_NO_TCC) == set(nb.TABELAS) | set(nb.FIGURAS)


@pytest.mark.parametrize("nome", sorted(nb.EXPERIMENTOS))
def test_experimento_e_um_subcomando_valido(nome):
    exp = nb.EXPERIMENTOS[nome]

    _parser().parse_args(list(exp.args))

    assert set(exp.tabelas) <= set(nb.TABELAS)
    assert set(exp.figuras) <= set(nb.FIGURAS)


@pytest.mark.parametrize("nome", sorted(nb.EXPERIMENTOS))
def test_arquivos_gravados_existem_em_results(nome):
    """Com os resultados versionados, o notebook com RODAR = False não roda nada."""
    faltando = [g for g in nb.EXPERIMENTOS[nome].gravados if not (RESULTS / g).exists()]
    assert not faltando


def test_toda_tabela_aparece_em_alguma_secao():
    mostradas = {t for exp in nb.EXPERIMENTOS.values() for t in exp.tabelas}
    # A tab_qualitativa é julgamento, não experimento: aparece só no índice.
    assert mostradas | {"tab_qualitativa"} == set(nb.TABELAS)


def test_sem_latex():
    assert nb.sem_latex("\\textit{Angle}") == "Angle"
    assert nb.sem_latex("0,873 $\\pm$ 0,045") == "0,873 ± 0,045"
    assert nb.sem_latex("$\\omega_{\\max}$ em $x_1$") == "ω_max em x_1"
    assert nb.sem_latex(3) == 3
