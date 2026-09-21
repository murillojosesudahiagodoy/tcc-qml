"""Testes dos artefatos de saída: CLI, tabelas e figuras (T13, T15, T9).

O que se protege aqui é a rastreabilidade: as tabelas derivadas têm que nascer
dos CSVs, e o `.tex` tem que ser colável no Overleaf sem edição manual.
"""

import pandas as pd
import pytest

from tccqml import cli
from tccqml.tabelas import tab_acuracia, tab_custo, tab_qualitativa, to_latex


@pytest.fixture
def resumo_falso() -> pd.DataFrame:
    """Um `resumo.csv` mínimo, com as colunas que os runners produzem."""
    linhas = []
    custos = {
        "angle": {"n_qubits": 2, "n_params_circuito": 12, "depth": 11,
                  "depth_encoding": 1, "gates_1q": 14, "gates_2q": 4,
                  "avaliacoes_por_gradiente": 24},
        "amplitude": {"n_qubits": 1, "n_params_circuito": 6, "depth": 8,
                      "depth_encoding": 2, "gates_1q": 8, "gates_2q": 0,
                      "avaliacoes_por_gradiente": 12},
    }
    for encoding, custo in custos.items():
        for dataset in ("xor", "moons"):
            linhas.append(
                {
                    "encoding": encoding,
                    "dataset": dataset,
                    "acc_teste_mean": 0.8,
                    "acc_teste_std": 0.02,
                    "acc_treino_mean": 0.81,
                    "acc_treino_std": 0.02,
                    "custo_final_mean": 0.5,
                    "custo_final_std": 0.01,
                    "epoca_90pct_mean": 7.0,
                    "epoca_90pct_std": 1.0,
                    **custo,
                }
            )
    return pd.DataFrame(linhas)


def test_cli_listar_nao_tem_lista_paralela(capsys):
    """Registrar em ENCODINGS basta: a CLI lê do registro, não de uma cópia."""
    from tccqml.embeddings import ENCODINGS

    cli.main(["listar"])
    saida = capsys.readouterr().out

    for nome in ENCODINGS:
        assert nome in saida


def test_cli_recusa_codificacao_invalida_listando_as_validas():
    with pytest.raises(SystemExit):
        cli.main(["treinar", "--encoding", "teleporte"])


def test_cli_treina_de_ponta_a_ponta(capsys):
    cli.main(
        ["treinar", "--encoding", "angle", "--dataset", "moons", "--epocas", "2",
         "--n-samples", "60"]
    )
    saida = capsys.readouterr().out

    assert "custo do circuito" in saida
    assert "acurácia" in saida


def test_latex_e_um_ambiente_table_completo():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    tex = to_latex(df, caption="Legenda.", label="tab:teste")

    assert tex.startswith("\\begin{table}")
    assert tex.rstrip().endswith("\\end{table}")
    for pedaco in ("\\caption{Legenda.}", "\\label{tab:teste}", "\\begin{tabular}", "Fonte:"):
        assert pedaco in tex


def test_latex_nao_exige_pacote_novo():
    """Só tabular e hline: nada de booktabs, que o preâmbulo pode não ter."""
    df = pd.DataFrame({"A": [1], "B": [2]})

    tex = to_latex(df, caption="x", label="tab:x")

    for proibido in ("\\toprule", "\\midrule", "\\bottomrule", "\\usepackage"):
        assert proibido not in tex


def test_tab_acuracia_sai_do_resumo(resumo_falso):
    tabela, tex = tab_acuracia(resumo_falso)

    assert next(iter(tabela.columns)) == "Codificação"
    assert "0.800" in tex and "0.020" in tex


def test_tab_custo_reporta_os_itens_da_etapa_9(resumo_falso):
    tabela, _ = tab_custo(resumo_falso)

    for coluna in ("Qubits", "$p$", "Prof. total", "Prof. codif.", "Portas 1q", "Portas 2q"):
        assert coluna in tabela.columns


def test_tab_qualitativa_cobre_as_quatro_codificacoes():
    tabela, tex = tab_qualitativa()

    assert len(tabela) == 4
    assert "Dificuldade" in tabela.columns
    assert "\\begin{table}" in tex


def test_figuras_tem_os_nomes_que_o_latex_espera(tmp_path):
    """Nome errado quebra o \\includegraphics sem dar erro no Python."""
    from tccqml.figuras import gerar_todas

    gerados = gerar_todas(out=tmp_path, verbose=False)
    nomes = {p.name for p in gerados}

    # Sem CSVs, só as figuras que não dependem de experimento são geradas.
    assert {"datasets.pdf", "espectro-reuploading.pdf"} <= nomes
    for caminho in gerados:
        assert caminho.exists() and caminho.stat().st_size > 0
