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
    tab_diagnostico,
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
                  "gates_total": 18, "gates_1q_encoding": 2, "gates_2q_encoding": 0,
                  "gates_total_encoding": 2,
                  "aval_derivadas_amostra": 24, "aval_gradiente_amostra": 25,
                  "n_aval_passo": 500, "n_shots_passo": 500_000},
        "amplitude": {"n_qubits": 1, "n_params_circuito": 6, "depth": 8,
                      "depth_encoding": 2, "gates_1q": 8, "gates_2q": 0,
                      "gates_total": 8, "gates_1q_encoding": 2, "gates_2q_encoding": 0,
                      "gates_total_encoding": 2,
                      "aval_derivadas_amostra": 12, "aval_gradiente_amostra": 13,
                      "n_aval_passo": 260, "n_shots_passo": 260_000},
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


def test_cli_treinar_salvar_grava_pesos_que_reproduzem_o_teste(tmp_path):
    """Os pesos gravados são os da época escolhida: refazem a acc_teste do resumo."""
    import numpy as np

    from tccqml import model
    from tccqml.data import load_dataset
    from tccqml.metrics import accuracy

    base = ["--out", str(tmp_path), "treinar", "--encoding", "reuploading",
            "--dataset", "moons", "--epocas", "2", "--n-samples", "60", "--salvar"]
    cli.main(base)
    nome = "treino_reuploading_moons_42_ep2_n60"
    assert (tmp_path / "metrics" / f"{nome}.csv").exists()
    resumo = pd.read_csv(tmp_path / "metrics" / f"{nome}_resumo.csv").iloc[0]
    pesos = np.load(tmp_path / "weights" / f"{nome}.npz")

    ds = load_dataset("moons", n_samples=60, seed=42)
    clf = model.build("reuploading", n_features=2, enc_kwargs={"R": 3})
    alpha = pesos["alpha"] if "alpha" in pesos.files else None
    previsto = model.prever(clf, pesos["weights"], alpha, float(pesos["bias"]), ds.X_test)
    assert accuracy(ds.y_test, previsto) == pytest.approx(resumo["acc_teste"])
    assert resumo["epoca_escolhida"] >= 1

    # Um eta fora do protocolo ganha sufixo e não sobrescreve o treino anterior.
    cli.main(base + ["--eta", "0.03"])
    assert (tmp_path / "weights" / f"{nome}.npz").exists()
    assert (tmp_path / "weights" / "treino_reuploading_moons_42_eta0.03_ep2_n60.npz").exists()


def test_latex_e_um_ambiente_table_completo():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    tex = to_latex(df, caption="Legenda.", label="tab:teste")

    assert tex.startswith("\\begin{table}")
    assert tex.rstrip().endswith("\\end{table}")
    for pedaco in ("\\caption{Legenda.}", "\\label{tab:teste}", "\\begin{tabular}", "Fonte:"):
        assert pedaco in tex


def test_latex_usa_booktabs_sem_carregar_pacote():
    """Regras do booktabs, como as tabelas do texto; nenhum \\usepackage no .tex."""
    df = pd.DataFrame({"A": [1], "B": [2]})

    tex = to_latex(df, caption="x", label="tab:x")

    for regra in ("\\toprule", "\\midrule", "\\bottomrule"):
        assert regra in tex
    assert "\\hline" not in tex and "\\usepackage" not in tex


def test_latex_com_legenda_curta_para_a_lista_de_tabelas():
    df = pd.DataFrame({"A": [1], "B": [2]})

    tex = to_latex(df, caption="Legenda longa.", label="tab:x", caption_curta="Curta")

    assert "\\caption[Curta]{Legenda longa.}" in tex


def test_latex_grupos_usam_cmidrule_lr():
    """`(lr)` é a opção do booktabs que encurta a regra dos dois lados, não a taxa."""
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})

    tex = to_latex(df, caption="x", label="tab:x", grupos=[("", 1), ("Grupo", 2)])

    assert "\\cmidrule(lr){2-3}" in tex


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

    for coluna in ("$p$", "Codif.: prof.", "Codif.: portas", "Codif.: CNOTs",
                   "Circuito: qubits", "Circuito: prof.", "Circuito: portas",
                   "Circuito: CNOTs"):
        assert coluna in tabela.columns


def test_tab_custo_separa_bloco_de_codificacao_do_circuito_completo(resumo_falso):
    tabela, tex = tab_custo(resumo_falso)
    angle = tabela.iloc[0]

    assert (angle["Codif.: prof."], angle["Codif.: portas"], angle["Codif.: CNOTs"]) == (1, 2, 0)
    assert (angle["Circuito: qubits"], angle["Circuito: portas"], angle["Circuito: CNOTs"]) == (2, 18, 4)
    assert "\\multicolumn{3}{c}{Bloco de codificação}" in tex
    assert "\\multicolumn{4}{c}{Circuito completo}" in tex


def test_tab_custo_mede_as_portas_de_um_resumo_antigo(resumo_falso):
    """Sem as colunas de portas no CSV, o circuito é medido de novo, sem treino."""
    antigo = resumo_falso.drop(
        columns=["gates_total", "gates_1q_encoding", "gates_2q_encoding", "gates_total_encoding"]
    )
    novo, _ = tab_custo(resumo_falso)
    medido, _ = tab_custo(antigo)

    pd.testing.assert_frame_equal(novo, medido)


def test_tab_custo_separa_avaliacoes_de_shots(resumo_falso):
    """2p só como comparação; o custo é (2p + 1)|B| avaliações e (2p + 1)|B|S shots."""
    tabela, tex = tab_custo(resumo_falso)
    angle = tabela.iloc[0]

    assert angle["Desloc. ($2p$)"] == 24
    assert angle["Aval./amostra"] == 25
    assert angle["Aval./passo"] == 500
    assert angle["\\textit{Shots}/passo"] == "500\\,000"
    assert "$(2p + 1)|B|S$" in tex


def test_tab_custo_recalcula_contagens_de_um_resumo_antigo(resumo_falso):
    """Um resumo.csv anterior à correção (2p + 1) não tem as colunas novas."""
    antigo = resumo_falso.drop(
        columns=["aval_derivadas_amostra", "aval_gradiente_amostra", "n_aval_passo", "n_shots_passo"]
    )
    novo, _ = tab_custo(resumo_falso)
    recalculado, _ = tab_custo(antigo)

    pd.testing.assert_frame_equal(novo, recalculado)


def test_tab_ablacao_tem_uma_coluna_por_conjunto():
    """A ablação roda nos três conjuntos: a queda não é só do xor."""
    ablacao = pd.DataFrame(
        [
            {"dataset": d, "ansatz": a, "seed": s, "acc_teste": v}
            for d in ("xor", "moons")
            for a, v in (("strongly_entangling", 0.9), ("local", 0.5))
            for s in (42, 43)
        ]
    )

    tabela, tex = tab_ablacao(ablacao)

    assert list(tabela.columns) == ["\\textit{Ansatz}", "\\textit{Xor}", "\\textit{Moons}"]
    assert "0,900 $\\pm$ 0,000" in tex and "0,500" in tex


def test_tab_diagnostico_sai_do_csv_com_o_protocolo_primeiro():
    """Uma linha por normalização, [0, pi] primeiro, e média +- desvio do CSV."""
    diagnostico = pd.DataFrame(
        [
            {"normalizacao": n, "dataset": d, "seed": s, "acc_teste": v + 0.1 * (s - 42)}
            for n, v in (("[-1,1]", 0.5), ("[0,pi]", 0.8))
            for d in ("xor", "circles")
            for s in (42, 43)
        ]
    )

    tabela, tex = tab_diagnostico(diagnostico)

    assert list(tabela.columns) == ["Normalização", "\\textit{Xor}", "\\textit{Circles}"]
    assert tabela["Normalização"].iloc[0].startswith("$[0, \\pi]$")
    assert tabela["\\textit{Xor}"].iloc[0] == "0,850 $\\pm$ 0,071"
    assert "\\label{tab:diagnostico}" in tex


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

    tabela, _ = tab_mesmo_p(varredura("R", {1: 6, 2: 12}), varredura("L_var", {1: 6, 2: 12}))
    assert list(tabela["$p$"]) == [6, 12]

    with pytest.raises(ValueError, match="p diferente"):
        tab_mesmo_p(varredura("R", {1: 6}), varredura("L_var", {1: 7}))


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


def test_figuras_sao_reproduziveis_byte_a_byte(tmp_path):
    """Regenerar sem mudança de conteúdo não pode mudar o PDF (sem data embutida)."""
    from tccqml.figuras import gerar_todas

    primeira = {p.name: p.read_bytes() for p in gerar_todas(out=tmp_path / "a", verbose=False)}
    segunda = {p.name: p.read_bytes() for p in gerar_todas(out=tmp_path / "b", verbose=False)}

    assert primeira and primeira == segunda
