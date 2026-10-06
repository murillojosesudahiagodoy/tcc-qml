"""Testes da verificação de sensibilidade da taxa de aprendizado.

O que se protege aqui é a honestidade da escolha: o lr é escolhido só pela
validação, com uma regra de empate fixa, e todas as codificações recebem o
mesmo orçamento de candidatos e sementes. Os treinos são substituídos por um
falso — o que importa é a contabilidade, não a otimização.
"""

import itertools

import pandas as pd
import pytest

from tccqml import experiments
from tccqml.config import PADRAO
from tccqml.experiments import (
    COLUNAS_SENSIBILIDADE,
    escolher_lr,
    resumir_sensibilidade,
    rodar_sensibilidade_lr,
)
from tccqml.tabelas import tab_sensibilidade


def _por_treino(acc_val: dict[float, list[float]], acc_teste: dict[float, list[float]]) -> pd.DataFrame:
    """Um `sensibilidade_lr.csv` sintético de um único (encoding, dataset)."""
    linhas = []
    for lr, vals in acc_val.items():
        for i, (val, teste) in enumerate(zip(vals, acc_teste[lr])):
            linhas.append(
                {
                    "encoding": "angle",
                    "dataset": "moons",
                    "lr": lr,
                    "seed": 42 + i,
                    "acc_treino": val,
                    "acc_val": val,
                    "acc_teste": teste,
                    "epoca_escolhida": 10,
                    "p": 12,
                    "n_qubits": 2,
                }
            )
    return pd.DataFrame(linhas)


def test_configuracao_nao_mexe_no_lr_do_protocolo():
    """Os candidatos são acrescentados; o lr da grade principal continua 0.1."""
    assert PADRAO.lr == 0.1
    assert PADRAO.lr_candidatos == (0.01, 0.03, 0.1, 0.3)
    assert PADRAO.lr in PADRAO.lr_candidatos
    assert PADRAO.lr_tolerancia_empate == 1e-9


# --------------------------------------------------------------------------
# (a) a escolha usa só a validação
# --------------------------------------------------------------------------


def test_escolha_do_lr_usa_so_a_validacao():
    """Melhor na validação: 0.03. Melhor no teste: 0.3. Tem que sair 0.03."""
    por_treino = _por_treino(
        acc_val={0.01: [0.70] * 3, 0.03: [0.90] * 3, 0.1: [0.80] * 3, 0.3: [0.60] * 3},
        acc_teste={0.01: [0.70] * 3, 0.03: [0.75] * 3, 0.1: [0.80] * 3, 0.3: [0.99] * 3},
    )

    escolha = escolher_lr(por_treino, lr_protocolo=0.1)

    assert escolha["lr_escolhido"].tolist() == [0.03]


def test_acc_teste_nao_influencia_a_escolha():
    """Embaralhar o teste por completo não pode mudar o lr escolhido."""
    por_treino = _por_treino(
        acc_val={0.01: [0.6, 0.7], 0.03: [0.8, 0.9], 0.1: [0.7, 0.8], 0.3: [0.5, 0.5]},
        acc_teste={0.01: [0.1, 0.1], 0.03: [0.2, 0.2], 0.1: [0.3, 0.3], 0.3: [0.4, 0.4]},
    )
    sem_teste = por_treino.drop(columns=["acc_teste"])

    assert escolher_lr(sem_teste, 0.1).equals(escolher_lr(por_treino, 0.1))


def test_media_e_nas_sementes_nao_no_melhor_treino():
    """0.3 tem o melhor treino isolado, mas 0.03 tem a maior MÉDIA de validação."""
    por_treino = _por_treino(
        acc_val={0.03: [0.85, 0.85, 0.85], 0.3: [1.0, 0.5, 0.5]},
        acc_teste={0.03: [0.8] * 3, 0.3: [0.8] * 3},
    )

    assert escolher_lr(por_treino, 0.1)["lr_escolhido"].tolist() == [0.03]


# --------------------------------------------------------------------------
# (b) regra de empate
# --------------------------------------------------------------------------


def test_empate_com_o_lr_do_protocolo_fica_com_o_protocolo():
    """0.03, 0.1 e 0.3 empatam: fica 0.1, mesmo não sendo o menor."""
    por_treino = _por_treino(
        acc_val={0.01: [0.5] * 2, 0.03: [0.9] * 2, 0.1: [0.9] * 2, 0.3: [0.9] * 2},
        acc_teste={lr: [0.5] * 2 for lr in (0.01, 0.03, 0.1, 0.3)},
    )

    assert escolher_lr(por_treino, 0.1)["lr_escolhido"].tolist() == [0.1]


def test_empate_sem_o_protocolo_fica_com_o_menor():
    """0.03 e 0.3 empatam acima de 0.1: fica o menor, 0.03."""
    por_treino = _por_treino(
        acc_val={0.01: [0.5] * 2, 0.03: [0.9] * 2, 0.1: [0.8] * 2, 0.3: [0.9] * 2},
        acc_teste={lr: [0.5] * 2 for lr in (0.01, 0.03, 0.1, 0.3)},
    )

    assert escolher_lr(por_treino, 0.1)["lr_escolhido"].tolist() == [0.03]


def test_empate_tolera_erro_de_ponto_flutuante():
    """Médias iguais na matemática, mas somadas em ordens diferentes, empatam."""
    por_treino = _por_treino(
        acc_val={0.03: [0.1, 0.2, 0.7], 0.1: [0.7, 0.1, 0.2]},
        acc_teste={0.03: [0.5] * 3, 0.1: [0.5] * 3},
    )

    assert escolher_lr(por_treino, 0.1)["lr_escolhido"].tolist() == [0.1]


def test_tolerancia_do_empate_e_a_do_protocolo():
    """1e-6 de diferença não empata com 1e-9; empata se a tolerância for maior."""
    por_treino = _por_treino(
        acc_val={0.03: [0.800001] * 2, 0.1: [0.8] * 2},
        acc_teste={0.03: [0.5] * 2, 0.1: [0.5] * 2},
    )

    assert escolher_lr(por_treino, 0.1)["lr_escolhido"].tolist() == [0.03]
    assert escolher_lr(por_treino, 0.1, tolerancia=1e-3)["lr_escolhido"].tolist() == [0.1]


def test_tolerancia_entre_medias_distintas_de_verdade():
    """Médias de 60 amostras x 5 sementes diferem em >= 1/300, nunca empatam."""
    assert 1 / 300 > 1000 * PADRAO.lr_tolerancia_empate


# --------------------------------------------------------------------------
# (c) o CSV tem todas as combinações
# --------------------------------------------------------------------------


@pytest.fixture
def treino_falso(monkeypatch):
    """Substitui `rodar_um` e registra com que protocolo cada treino rodou."""
    chamadas = []

    def falso(encoding, dataset, seed, protocolo=PADRAO, **kwargs):
        chamadas.append((encoding, dataset, protocolo.lr, seed, protocolo))
        # A validação favorece lr = 0.03 e o teste favorece lr = 0.3: se a
        # escolha olhasse o teste, o resumo mostraria 0.3.
        acc_val = 0.9 if protocolo.lr == 0.03 else 0.7
        acc_teste = 0.95 if protocolo.lr == 0.3 else 0.6
        resumo = {
            "acc_treino": 0.8,
            "acc_val": acc_val,
            "acc_teste": acc_teste,
            "epoca_escolhida": 5,
            "n_params_circuito": 6 if encoding == "amplitude" else 12,
            "n_qubits": 1 if encoding == "amplitude" else 2,
        }
        return None, resumo, None

    monkeypatch.setattr(experiments, "rodar_um", falso)
    return chamadas


def test_csv_tem_todas_as_combinacoes(tmp_path, treino_falso):
    rodar_sensibilidade_lr(out=tmp_path, verbose=False)

    csv = pd.read_csv(tmp_path / "metrics" / "sensibilidade_lr.csv")
    esperado = set(
        itertools.product(
            PADRAO.encodings_grade, PADRAO.datasets_grade, PADRAO.lr_candidatos, PADRAO.sementes
        )
    )
    obtido = set(csv[["encoding", "dataset", "lr", "seed"]].itertuples(index=False, name=None))

    assert list(csv.columns) == list(COLUNAS_SENSIBILIDADE)
    assert len(csv) == len(esperado) == 4 * 3 * 4 * 5
    assert obtido == esperado


def test_so_o_lr_varia_entre_os_treinos(tmp_path, treino_falso):
    """Mesmo protocolo da grade (L_reup, épocas, lotes...), exceto o lr."""
    rodar_sensibilidade_lr(out=tmp_path, verbose=False)

    for _, _, lr, _, protocolo in treino_falso:
        assert protocolo.lr == lr
        assert protocolo.L_reup == PADRAO.L_reup
        assert protocolo.epocas == PADRAO.epocas
        assert protocolo.batch_size == PADRAO.batch_size
        assert protocolo.n_layers == PADRAO.n_layers


def test_resumo_escolhe_pela_validacao_e_traz_o_protocolo_ao_lado(tmp_path, treino_falso):
    rodar_sensibilidade_lr(out=tmp_path, verbose=False)

    resumo = pd.read_csv(tmp_path / "metrics" / "resumo_sensibilidade_lr.csv")

    assert len(resumo) == len(PADRAO.encodings_grade) * len(PADRAO.datasets_grade)
    assert (resumo["lr_escolhido"] == 0.03).all()
    assert (resumo["lr_protocolo"] == PADRAO.lr).all()
    assert (resumo["acc_val_mean_escolhido"] == 0.9).all()
    assert (resumo["acc_teste_mean_escolhido"] == 0.6).all()
    assert (resumo["acc_val_mean_protocolo"] == 0.7).all()
    assert (resumo["n_sementes"] == len(PADRAO.sementes)).all()
    for coluna in ("acc_val_std_escolhido", "acc_teste_std_escolhido",
                   "acc_val_std_protocolo", "acc_teste_std_protocolo"):
        assert coluna in resumo.columns


def test_runner_usa_a_tolerancia_do_protocolo(tmp_path, treino_falso):
    """Com tolerância 0.5 tudo empata (0.9 contra 0.7) e fica o lr do protocolo."""
    from dataclasses import replace

    rodar_sensibilidade_lr(
        replace(PADRAO, lr_tolerancia_empate=0.5), out=tmp_path, verbose=False
    )
    resumo = pd.read_csv(tmp_path / "metrics" / "resumo_sensibilidade_lr.csv")

    assert (resumo["lr_escolhido"] == PADRAO.lr).all()


def test_recusa_lr_do_protocolo_fora_dos_candidatos(tmp_path, treino_falso):
    """Sem o lr da grade entre os candidatos, não há coluna de comparação."""
    from dataclasses import replace

    with pytest.raises(ValueError, match="fora de lr_candidatos"):
        rodar_sensibilidade_lr(replace(PADRAO, lr=0.2), out=tmp_path, verbose=False)
    assert treino_falso == []


# --------------------------------------------------------------------------
# tabela e figura
# --------------------------------------------------------------------------


def test_tab_sensibilidade_uma_linha_por_codificacao():
    por_treino = pd.concat(
        [
            _por_treino(
                acc_val={0.03: [0.9, 0.8], 0.1: [0.7, 0.7]},
                acc_teste={0.03: [0.85, 0.85], 0.1: [0.65, 0.65]},
            ).assign(encoding=enc, dataset=ds)
            for enc in ("angle", "amplitude")
            for ds in ("xor", "moons")
        ]
    )
    resumo = resumir_sensibilidade(por_treino, lr_protocolo=0.1)

    tabela, tex = tab_sensibilidade(resumo, candidatos=[0.03, 0.1])

    assert len(tabela) == 2
    assert list(tabela.columns) == [
        "Codificação",
        "XOR (protocolo)",
        "XOR (escolhido)",
        "\\textit{Moons} (protocolo)",
        "\\textit{Moons} (escolhido)",
    ]
    assert tabela.iloc[0]["XOR (protocolo)"] == "0,650 $\\pm$ 0,000"
    assert tabela.iloc[0]["XOR (escolhido)"] == "0,850 $\\pm$ 0,000 (0,03)"
    assert "validação" in tex
    assert "sobre duas sementes" in tex


def test_figura_de_sensibilidade(tmp_path, treino_falso):
    from tccqml.figuras import gerar_todas

    rodar_sensibilidade_lr(out=tmp_path, verbose=False)
    nomes = {p.name for p in gerar_todas(out=tmp_path, verbose=False)}

    assert "sensibilidade-lr.pdf" in nomes
