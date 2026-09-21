"""Runner dos experimentos — a comparação da Etapa 10 do roteiro.

Aqui mora a grade que produz os números do Capítulo 4. Tudo roda com o MESMO
protocolo (`config.Protocolo`): a codificação é a única coisa que varia, que é
a condição para atribuir a diferença de desempenho à codificação e não ao
otimizador, ao ansatz ou aos dados.

O treino usa `default.qubit` com retropropagação. O parameter-shift NÃO é
executado — ele é contabilizado analiticamente em `circuit_stats.py`
(Seção 2.3.5.1). As colunas de custo de gradiente que saem nos CSVs são,
portanto, o custo que este mesmo treino teria em hardware real.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from tccqml import model
from tccqml.circuit_stats import stats
from tccqml.config import PADRAO, Protocolo
from tccqml.data import load_dataset
from tccqml.train import treinar

COLUNAS_CUSTO = (
    "n_qubits",
    "depth",
    "depth_encoding",
    "gates_1q",
    "gates_2q",
    "n_params_ansatz",
    "n_params_encoding",
    "n_params_circuito",
    "avaliacoes_por_gradiente",
    "n_execucoes_hardware",
)


def _enc_kwargs(encoding: str, L_reup: int) -> dict | None:
    return {"L_reup": L_reup} if encoding == "reuploading" else None


def rodar_um(
    encoding: str,
    dataset: str,
    seed: int,
    protocolo: Protocolo = PADRAO,
    ansatz: str | None = None,
    L_reup: int | None = None,
    feature_range: tuple[float, float] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict, object]:
    """Um treino completo. Devolve (histórico anotado, resumo, resultado).

    O histórico já sai com as colunas de custo do circuito repetidas em toda
    linha: desempenho e custo lado a lado é a tese do trabalho, e deixa o CSV
    autossuficiente para as tabelas do Capítulo 4.
    """
    L_reup = protocolo.L_reup if L_reup is None else L_reup
    ds = load_dataset(
        dataset,
        n_samples=protocolo.n_samples,
        noise=protocolo.noise,
        test_size=protocolo.test_size,
        seed=seed,
        feature_range=feature_range or protocolo.feature_range,
    )
    clf = model.build(
        encoding,
        n_features=ds.n_features,
        n_layers=protocolo.n_layers,
        ansatz=ansatz or protocolo.ansatz,
        enc_kwargs=_enc_kwargs(encoding, L_reup),
    )
    custo = stats(
        clf,
        ds.X_train[0],
        batch_size=protocolo.batch_efetivo,
        shots=protocolo.shots,
    )

    t0 = time.perf_counter()
    r = treinar(
        clf,
        ds,
        epocas=protocolo.epocas,
        batch_size=protocolo.batch_size,
        lr=protocolo.lr,
        seed=seed,
        verbose=verbose,
    )
    segundos = time.perf_counter() - t0

    hist = r.historico.copy()
    hist.insert(0, "seed", seed)
    hist.insert(0, "dataset", dataset)
    hist.insert(0, "encoding", encoding)
    hist["ansatz"] = clf.ansatz
    hist["L_reup"] = L_reup if encoding == "reuploading" else np.nan
    for coluna in COLUNAS_CUSTO:
        hist[coluna] = custo[coluna]

    resumo = {
        "encoding": encoding,
        "dataset": dataset,
        "seed": seed,
        "ansatz": clf.ansatz,
        "L_reup": L_reup if encoding == "reuploading" else np.nan,
        "acc_treino": r.acc_treino,
        "acc_teste": r.acc_teste,
        "custo_final": float(hist["custo"].iloc[-1]),
        "epoca_90pct": epoca_para_90pct(hist),
        "segundos": segundos,
        **{c: custo[c] for c in COLUNAS_CUSTO},
    }
    return hist, resumo, r


def epoca_para_90pct(hist: pd.DataFrame, coluna: str = "acc_treino") -> int:
    """Primeira época que atinge 90% da acurácia final — medida de velocidade.

    Convergir rápido e convergir alto são coisas diferentes; esta coluna isola a
    primeira (é o que a `tab_convergencia` reporta).
    """
    alvo = 0.9 * float(hist[coluna].iloc[-1])
    atingiu = hist.index[hist[coluna] >= alvo]
    return int(hist.loc[atingiu[0], "epoca"]) if len(atingiu) else int(hist["epoca"].iloc[-1])


def _salvar_pesos(destino: Path, nome: str, r) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    dados = {"weights": np.asarray(r.weights, dtype=float), "bias": float(r.bias)}
    if r.alpha is not None:
        dados["alpha"] = np.asarray(r.alpha, dtype=float)
    np.savez(destino / f"{nome}.npz", **dados)


def rodar_grade(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    encodings: tuple[str, ...] | None = None,
    datasets: tuple[str, ...] | None = None,
    sementes: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """A grade principal: codificações x datasets x sementes.

    No protocolo padrão são 4 x 3 x 5 = 60 treinos. Os pesos finais de cada
    treino vão para disco, para `figuras.py` desenhar as fronteiras sem
    retreinar.
    """
    out = Path(out or protocolo.out)
    encodings = encodings or protocolo.encodings_grade
    datasets = datasets or protocolo.datasets_grade
    sementes = sementes or protocolo.sementes

    historicos, resumos = [], []
    total = len(encodings) * len(datasets) * len(sementes)
    i = 0
    t_inicio = time.perf_counter()

    for encoding in encodings:
        for dataset in datasets:
            for seed in sementes:
                i += 1
                hist, resumo, r = rodar_um(encoding, dataset, seed, protocolo)
                historicos.append(hist)
                resumos.append(resumo)
                _salvar_pesos(out / "weights", f"{encoding}_{dataset}_{seed}", r)
                if verbose:
                    decorrido = time.perf_counter() - t_inicio
                    restante = decorrido / i * (total - i)
                    print(
                        f"[{i:3d}/{total}] {encoding:12s} {dataset:8s} seed={seed} "
                        f"teste={resumo['acc_teste']:.3f}  "
                        f"({resumo['segundos']:.1f}s, faltam ~{restante / 60:.1f} min)"
                    )

    comparacao = pd.concat(historicos, ignore_index=True)
    por_treino = pd.DataFrame(resumos)

    metrics = out / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    comparacao.to_csv(metrics / "comparacao.csv", index=False)
    por_treino.to_csv(metrics / "por_treino.csv", index=False)
    resumir(por_treino).to_csv(metrics / "resumo.csv", index=False)

    if verbose:
        print(f"\ngrade concluída em {(time.perf_counter() - t_inicio) / 60:.1f} min")
        print(f"escrito em {metrics}")
    return comparacao


def resumir(por_treino: pd.DataFrame) -> pd.DataFrame:
    """Média e desvio entre sementes, por (codificação, dataset).

    O desvio não é decoração: com 5 sementes, uma diferença de acurácia menor
    que o desvio não sustenta afirmação nenhuma no Capítulo 4.
    """
    agregacoes = {
        "acc_teste": ["mean", "std"],
        "acc_treino": ["mean", "std"],
        "custo_final": ["mean", "std"],
        "epoca_90pct": ["mean", "std"],
        "segundos": ["mean"],
    }
    chaves = ["encoding", "dataset"]
    resumo = por_treino.groupby(chaves, as_index=False).agg(agregacoes)
    resumo.columns = ["_".join(c).rstrip("_") for c in resumo.columns]

    # As colunas de custo são estáticas dentro de cada (enc, dataset): basta a
    # primeira ocorrência, e assim o resumo já serve à tab_custo sem novo join.
    custo = por_treino.groupby(chaves, as_index=False)[list(COLUNAS_CUSTO)].first()
    return resumo.merge(custo, on=chaves)


def rodar_varredura_L(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    valores: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] | None = None,
    sementes: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Varredura do número de blocos do re-uploading.

    Testa duas das cinco previsões da Seção 2.5.9: a 1 (ganho no `moons` a
    partir de L = 2) e a 5 (saturação da acurácia enquanto o custo sobe
    linearmente — as colunas de custo saem junto para isso ser verificável na
    mesma tabela).
    """
    out = Path(out or protocolo.out)
    valores = valores or protocolo.L_reup_varredura
    datasets = datasets or protocolo.datasets_grade
    sementes = sementes or protocolo.sementes

    linhas = []
    total = len(valores) * len(datasets) * len(sementes)
    i = 0
    for L in valores:
        for dataset in datasets:
            for seed in sementes:
                i += 1
                _, resumo, _ = rodar_um(
                    "reuploading", dataset, seed, protocolo, L_reup=L
                )
                linhas.append(resumo)
                if verbose:
                    print(
                        f"[{i:3d}/{total}] L={L} {dataset:8s} seed={seed} "
                        f"teste={resumo['acc_teste']:.3f}"
                    )

    varredura = pd.DataFrame(linhas)
    metrics = out / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    varredura.to_csv(metrics / "varredura_L.csv", index=False)
    if verbose:
        print(f"escrito em {metrics / 'varredura_L.csv'}")
    return varredura


def rodar_ablacao(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    dataset: str = "xor",
    encoding: str = "angle",
    sementes: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Ablação do entrelaçamento do ansatz (Previsão 3, Eq. 2.83).

    Roda a mesma configuração com e sem CNOTs no ansatz. Sem entrelaçamento o
    modelo passa a enxergar só o primeiro atributo e a acurácia no XOR cai para
    o nível do acaso.
    """
    out = Path(out or protocolo.out)
    sementes = sementes or protocolo.sementes
    linhas = []
    for ansatz in ("strongly_entangling", "local"):
        for seed in sementes:
            _, resumo, _ = rodar_um(encoding, dataset, seed, protocolo, ansatz=ansatz)
            linhas.append(resumo)
            if verbose:
                print(f"{ansatz:20s} seed={seed} teste={resumo['acc_teste']:.3f}")

    ablacao = pd.DataFrame(linhas)
    metrics = out / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    ablacao.to_csv(metrics / "ablacao_entrelacamento.csv", index=False)
    if verbose:
        medias = ablacao.groupby("ansatz")["acc_teste"].mean()
        print("\nacurácia de teste média:")
        print(medias.to_string())
    return ablacao


def diagnostico_amplitude(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    datasets: tuple[str, ...] | None = None,
    sementes: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Diagnóstico do amplitude: normalização [0, pi] contra [-1, 1].

    O amplitude encoding descarta a norma e só enxerga a DIREÇÃO do vetor
    (Seção 2.4.4.1). Como `data.py` normaliza para [0, pi], todos os pontos
    caem no primeiro quadrante e o ângulo entre dois vetores quaisquer fica
    espremido em [0, pi/2] — o que pode arruinar o desempenho por um motivo que
    não é da codificação.

    Este runner mede as duas normalizações e reporta as duas. A escolha de qual
    usar no texto é do autor, não do código; o resto do trabalho continua em
    [0, pi], que é o intervalo que o angle encoding exige.
    """
    out = Path(out or protocolo.out)
    datasets = datasets or protocolo.datasets_grade
    sementes = sementes or protocolo.sementes

    faixas = {"[0,pi]": protocolo.feature_range, "[-1,1]": (-1.0, 1.0)}
    linhas = []
    for rotulo, faixa in faixas.items():
        for dataset in datasets:
            for seed in sementes:
                _, resumo, _ = rodar_um(
                    "amplitude", dataset, seed, protocolo, feature_range=faixa
                )
                resumo["normalizacao"] = rotulo
                linhas.append(resumo)
                if verbose:
                    print(
                        f"{rotulo:8s} {dataset:8s} seed={seed} "
                        f"teste={resumo['acc_teste']:.3f}"
                    )

    diag = pd.DataFrame(linhas)
    metrics = out / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    diag.to_csv(metrics / "diagnostico_amplitude.csv", index=False)
    if verbose:
        print("\nacurácia de teste média por normalização e dataset:")
        print(
            diag.groupby(["normalizacao", "dataset"])["acc_teste"]
            .mean()
            .unstack()
            .round(3)
            .to_string()
        )
    return diag
