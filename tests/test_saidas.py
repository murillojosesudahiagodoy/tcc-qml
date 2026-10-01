"""Testes dos artefatos de saída: CLI, tabelas e figuras.

O que se protege aqui é a rastreabilidade: as tabelas derivadas têm que nascer
dos CSVs, e o `.tex` tem que ser colável no Overleaf sem edição manual.
"""

import pandas as pd
import pytest

from tccqml import cli
from tccqml.tabelas import (
    tab_ablacao,
    tab_acuracia,
    tab_custo,
    tab_mesmo_p,
    tab_qualitativa,
    to_latex,
)


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
                    "acc_teste_count": 5,
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
    # Vírgula decimal, como no texto (ABNT): o .tex é colado sem edição.
    assert "0,800" in tex and "0,020" in tex
    assert "0.800" not in tex

    # A nota de qubits e p também sai de cada linha do resumo, não de uma frase
    # fixa: mudar p de uma codificação tem que mudar a nota.
    assert "\\textit{Angle}, 2 e 12" in tex
    assert "\\textit{Amplitude}, 1 e 6" in tex
    maior = resumo_falso.assign(
        n_params_circuito=resumo_falso["n_params_circuito"].where(
            resumo_falso["encoding"] != "angle", 18
        )
    )
    _, tex_maior = tab_acuracia(maior)
    assert "\\textit{Angle}, 2 e 18" in tex_maior


def test_legenda_conta_as_sementes_do_csv(resumo_falso):
    """O número de sementes da legenda sai do CSV, não de um literal.

    Por extenso até dez, no padrão ABNT das tabelas já escritas.
    """
    _, tex = tab_acuracia(resumo_falso)
    assert "sobre cinco sementes" in tex

    doze = resumo_falso.assign(acc_teste_count=12)
    _, tex_doze = tab_acuracia(doze)
    assert "sobre 12 sementes" in tex_doze

    _, tex_sem_contagem = tab_acuracia(resumo_falso.drop(columns=["acc_teste_count"]))
    assert "sobre as sementes" in tex_sem_contagem


def test_tab_custo_reporta_os_itens_da_etapa_9(resumo_falso):
    tabela, _ = tab_custo(resumo_falso)

    for coluna in ("Qubits", "$p$", "Prof. total", "Prof. codif.", "Portas 1q", "Portas 2q"):
        assert coluna in tabela.columns


def test_tab_ablacao_tem_uma_coluna_por_conjunto():
    """A ablação roda nos três conjuntos: a queda não é só do XOR."""
    ablacao = pd.DataFrame(
        [
            {"dataset": d, "ansatz": a, "seed": s, "acc_teste": v}
            for d in ("xor", "moons")
            for a, v in (("strongly_entangling", 0.9), ("local", 0.5))
            for s in (42, 43)
        ]
    )

    tabela, tex = tab_ablacao(ablacao)

    assert list(tabela.columns) == ["\\textit{Ansatz}", "XOR", "\\textit{Moons}"]
    assert "0,900 $\\pm$ 0,000" in tex and "0,500" in tex


def test_tab_mesmo_p_recusa_p_diferente():
    """A comparação só vale entre modelos com o mesmo p; se não, é erro."""

    def varredura(coluna: str, p_por_k: dict[int, int]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {coluna: k, "dataset": "moons", "acc_teste": 0.8, "n_params_circuito": p}
                for k, p in p_por_k.items()
                for _ in range(2)
            ]
        )

    tabela, _ = tab_mesmo_p(varredura("L_reup", {1: 6, 2: 12}), varredura("n_layers", {1: 6, 2: 12}))
    assert list(tabela["$p$"]) == [6, 12]

    with pytest.raises(ValueError, match="p diferente"):
        tab_mesmo_p(varredura("L_reup", {1: 6}), varredura("n_layers", {1: 7}))


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
