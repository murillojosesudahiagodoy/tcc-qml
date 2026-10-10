"""Runner dos experimentos — a comparação entre codificações.

Aqui mora a grade que produz os números do Capítulo 4. Tudo roda com o MESMO
protocolo (`config.Protocolo`): a codificação é a única coisa que varia, que é
a condição para atribuir a diferença de desempenho à codificação e não ao
otimizador, ao ansatz ou aos dados.

O treino usa `default.qubit` com retropropagação. O parameter-shift NÃO é
executado — ele é contabilizado analiticamente em `circuit_stats.py`
(Cap. 2, "regra de deslocamento de parâmetro"). As colunas de custo de
gradiente que saem nos CSVs são,
portanto, o custo que este mesmo treino teria em hardware real.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tccqml import model
from tccqml.circuit_stats import stats
from tccqml.config import PADRAO, Protocolo
from tccqml.data import load_dataset
from tccqml.train import treinar

# Recursos do circuito completo e, separadamente, do bloco de codificação
# (Cap. 3, "Métricas"): qubits, profundidade, portas totais e portas de dois qubits.
COLUNAS_CUSTO = (
    "n_qubits",
    "depth",
    "gates_1q",
    "gates_2q",
    "gates_total",
    "depth_encoding",
    "gates_1q_encoding",
    "gates_2q_encoding",
    "gates_total_encoding",
    "n_params_ansatz",
    "n_params_encoding",
    "n_params_circuito",
    "aval_derivadas_amostra",
    "aval_gradiente_amostra",
    "n_aval_passo",
    "n_shots_passo",
)


def _enc_kwargs(encoding: str, R: int) -> dict | None:
    return {"R": R} if encoding == "reuploading" else None


def rodar_um(
    encoding: str,
    dataset: str,
    seed: int,
    protocolo: Protocolo = PADRAO,
    ansatz: str | None = None,
    R: int | None = None,
    feature_range: tuple[float, float] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict, object]:
    """Um treino completo. Devolve (histórico anotado, resumo, resultado).

    O histórico já sai com as colunas de custo do circuito repetidas em toda
    linha, o que deixa o CSV autossuficiente para as tabelas do Capítulo 4.
    """
    R = protocolo.R if R is None else R
    ds = load_dataset(
        dataset,
        n_samples=protocolo.n_samples,
        noise=protocolo.noise,
        val_size=protocolo.val_size,
        test_size=protocolo.test_size,
        seed=seed,
        feature_range=feature_range or protocolo.feature_range,
    )
    clf = model.build(
        encoding,
        n_features=ds.n_features,
        L_var=protocolo.L_var,
        ansatz=ansatz or protocolo.ansatz,
        enc_kwargs=_enc_kwargs(encoding, R),
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
        eta=protocolo.eta,
        seed=seed,
        verbose=verbose,
    )
    segundos = time.perf_counter() - t0

    hist = r.historico.copy()
    hist.insert(0, "seed", seed)
    hist.insert(0, "dataset", dataset)
    hist.insert(0, "encoding", encoding)
    hist["ansatz"] = clf.ansatz
    hist["R"] = R if encoding == "reuploading" else np.nan
    for coluna in COLUNAS_CUSTO:
        hist[coluna] = custo[coluna]

    resumo = {
        "encoding": encoding,
        "dataset": dataset,
        "seed": seed,
        "ansatz": clf.ansatz,
        "R": R if encoding == "reuploading" else np.nan,
        # Só faz diferença nas codificações que aplicam o ansatz depois do
        # bloco de dados; no re-uploading quem dita as camadas é `R`.
        "L_var": protocolo.L_var if encoding != "reuploading" else np.nan,
        # As três acurácias são do modelo da época escolhida pela validação;
        # `acc_teste` é a única medida feita no teste.
        "acc_treino": r.acc_treino,
        "acc_val": r.acc_val,
        "acc_teste": r.acc_teste,
        "epoca_escolhida": r.epoca_escolhida,
        "custo_final": float(hist["custo"].iloc[-1]),
        "epoca_90pct": epoca_para_90pct(hist),
        "segundos": segundos,
        **{c: custo[c] for c in COLUNAS_CUSTO},
    }
    return hist, resumo, r


def epoca_para_90pct(hist: pd.DataFrame, coluna: str = "acc_treino") -> int:
    """Primeira época em que a acurácia de treino atinge 90% do valor final.

    Coluna auxiliar de `por_treino.csv` e `resumo.csv`; não é usada em
    nenhuma tabela nem no texto (a antiga `tab_convergencia` foi retirada).
    Fica no CSV para que os resultados versionados continuem reproduzíveis
    coluna a coluna.
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
                        f"val={resumo['acc_val']:.3f} teste={resumo['acc_teste']:.3f} "
                        f"(época {resumo['epoca_escolhida']})  "
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
        # `count` deixa o número de sementes legível no CSV, para a legenda da
        # tab_acuracia não precisar repetir um número escrito à mão.
        "acc_teste": ["mean", "std", "count"],
        "acc_val": ["mean", "std", "count"],
        "acc_treino": ["mean", "std"],
        "epoca_escolhida": ["mean", "std"],
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


def rodar_varredura_R(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    valores: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] | None = None,
    sementes: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Varredura do número de repetições R do re-uploading (`R`).

    Testa as hipóteses H1 (no `moons`, o re-uploading melhora em relação ao
    angle a partir de R = 2; o limiar foi registrado, mas não demonstrado para
    este circuito, e a comparação é com o angle de mesmo p) e H5 (o ganho para
    de crescer enquanto o custo sobe linearmente — as colunas de custo saem
    junto para isso ser verificável na mesma tabela). Cap. 3, "Previsões e
    hipóteses".
    """
    out = Path(out or protocolo.out)
    valores = valores or protocolo.R_varredura
    datasets = datasets or protocolo.datasets_grade
    sementes = sementes or protocolo.sementes

    linhas = []
    total = len(valores) * len(datasets) * len(sementes)
    i = 0
    for R in valores:
        for dataset in datasets:
            for seed in sementes:
                i += 1
                _, resumo, _ = rodar_um(
                    "reuploading", dataset, seed, protocolo, R=R
                )
                linhas.append(resumo)
                if verbose:
                    print(
                        f"[{i:3d}/{total}] R={R} {dataset:8s} seed={seed} "
                        f"teste={resumo['acc_teste']:.3f}"
                    )

    varredura = pd.DataFrame(linhas)
    metrics = out / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    varredura.to_csv(metrics / "varredura_R.csv", index=False)
    if verbose:
        print(f"escrito em {metrics / 'varredura_R.csv'}")
    return varredura


def rodar_varredura_L_var(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    valores: tuple[int, ...] | None = None,
    datasets: tuple[str, ...] | None = None,
    sementes: tuple[int, ...] | None = None,
    encoding: str = "angle",
    verbose: bool = True,
) -> pd.DataFrame:
    """Varredura do número de camadas do ansatz numa codificação fixa.

    É o controle da varredura de R. No re-uploading, aumentar R aumenta ao
    mesmo tempo o suporte de frequências e o número de parâmetros. No angle,
    aumentar `L_var` (L_var) aumenta só os parâmetros: o suporte continua
    Omega = {-1, 0, 1} por atributo. Com dois qubits os dois têm p = 6 por
    camada, então angle com L_var = k e re-uploading com R = k têm o mesmo p,
    e a diferença de acurácia entre eles isola a reinserção dos dados do
    número de parâmetros. A profundidade não é a mesma: o re-uploading
    acrescenta R blocos de dados.

    Sai do protocolo congelado de propósito (`L_var` muda) e por isso
    grava num CSV próprio, que nunca entra na comparação principal.
    """
    out = Path(out or protocolo.out)
    valores = valores or protocolo.L_var_varredura
    datasets = datasets or protocolo.datasets_grade
    sementes = sementes or protocolo.sementes

    linhas = []
    total = len(valores) * len(datasets) * len(sementes)
    i = 0
    for L_var in valores:
        variante = replace(protocolo, L_var=L_var)
        for dataset in datasets:
            for seed in sementes:
                i += 1
                _, resumo, _ = rodar_um(encoding, dataset, seed, variante)
                linhas.append(resumo)
                if verbose:
                    print(
                        f"[{i:3d}/{total}] {encoding} L_var={L_var} "
                        f"{dataset:8s} seed={seed} teste={resumo['acc_teste']:.3f}"
                    )

    varredura = pd.DataFrame(linhas)
    metrics = out / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    varredura.to_csv(metrics / "varredura_L_var.csv", index=False)
    if verbose:
        print(f"escrito em {metrics / 'varredura_L_var.csv'}")
    return varredura


def rodar_ablacao(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    datasets: tuple[str, ...] | None = None,
    encoding: str = "angle",
    sementes: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Ablação do entrelaçamento do ansatz (hipótese H3).

    Roda a mesma configuração com e sem CNOTs no ansatz. Sem entrelaçamento o
    estado é produto e <Z_0> só enxerga o primeiro atributo (Cap. 2,
    "Entrelaçamento e termos cruzados"). Isso
    derruba QUALQUER conjunto que precise do segundo atributo, não só o xor —
    por isso a ablação roda nos três conjuntos por padrão: a queda em moons e
    circles é o controle que mostra o que a ablação de fato mede (dependência
    do entrelaçamento dado o observável local Z_0).
    """
    out = Path(out or protocolo.out)
    datasets = datasets or protocolo.datasets_grade
    sementes = sementes or protocolo.sementes
    linhas = []
    for dataset in datasets:
        for ansatz in ("strongly_entangling", "local"):
            for seed in sementes:
                _, resumo, _ = rodar_um(encoding, dataset, seed, protocolo, ansatz=ansatz)
                linhas.append(resumo)
                if verbose:
                    print(
                        f"{dataset:8s} {ansatz:20s} seed={seed} "
                        f"teste={resumo['acc_teste']:.3f}"
                    )

    ablacao = pd.DataFrame(linhas)
    metrics = out / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    ablacao.to_csv(metrics / "ablacao_entrelacamento.csv", index=False)
    if verbose:
        medias = ablacao.groupby(["dataset", "ansatz"])["acc_teste"].mean().unstack()
        print("\nacurácia de teste média:")
        print(medias.round(3).to_string())
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
    (Cap. 2, "Amplitude encoding"). Como `data.py` normaliza para [0, pi],
    todos os pontos caem no primeiro quadrante e o ângulo entre dois vetores
    quaisquer fica em [0, pi/2], e a fronteira, formada por direções que partem
    da origem, só separa setores angulares desse quadrante.

    Este runner mede as duas normalizações e reporta as duas. A comparação
    principal mantém [0, pi] para todas as codificações, que é a escala
    escolhida para o angle encoding (escala e localidade, não uma exigência).
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


# Colunas por treino de `sensibilidade_eta.csv`. `p` é `n_params_circuito`
# (sem o viés), o mesmo `p` das contagens de custo.
COLUNAS_SENSIBILIDADE = (
    "encoding",
    "dataset",
    "eta",
    "seed",
    "acc_treino",
    "acc_val",
    "acc_teste",
    "epoca_escolhida",
    "p",
    "n_qubits",
)


def rodar_sensibilidade_eta(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    encodings: tuple[str, ...] | None = None,
    datasets: tuple[str, ...] | None = None,
    sementes: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Verificação de sensibilidade da taxa de aprendizado.

    A grade principal usa o mesmo `eta` para todas as codificações. Se uma
    delas fica para trás, falta saber se é a representação que não alcança a
    fronteira ou só o otimizador que não chegou lá com aquele passo. Esta
    verificação separa as duas coisas: repete o treino da grade (mesmo split,
    mesma inicialização, mesma escolha de época pela validação, o mesmo
    `R` no re-uploading) variando APENAS `eta` sobre `eta_candidatos`.

    É um experimento à parte: grava em CSVs próprios e não toca nos da
    comparação principal, que continua com `protocolo.eta`. Todas as
    codificações recebem o mesmo orçamento de candidatos e sementes, e a
    escolha do `eta` olha só a validação (`escolher_eta`) — o teste aparece no
    resumo apenas como medida do eta já escolhido.
    """
    if not any(np.isclose(protocolo.eta, c) for c in protocolo.eta_candidatos):
        raise ValueError(
            f"eta do protocolo ({protocolo.eta}) fora de eta_candidatos: "
            "a comparação com o eta da grade principal ficaria sem referência"
        )
    out = Path(out or protocolo.out)
    encodings = encodings or protocolo.encodings_grade
    datasets = datasets or protocolo.datasets_grade
    sementes = sementes or protocolo.sementes

    linhas = []
    total = len(encodings) * len(datasets) * len(protocolo.eta_candidatos) * len(sementes)
    i = 0
    t_inicio = time.perf_counter()
    for encoding in encodings:
        for dataset in datasets:
            for eta in protocolo.eta_candidatos:
                variante = replace(protocolo, eta=eta)
                for seed in sementes:
                    i += 1
                    _, resumo, _ = rodar_um(encoding, dataset, seed, variante)
                    linhas.append(
                        {
                            "encoding": encoding,
                            "dataset": dataset,
                            "eta": eta,
                            "seed": seed,
                            "acc_treino": resumo["acc_treino"],
                            "acc_val": resumo["acc_val"],
                            "acc_teste": resumo["acc_teste"],
                            "epoca_escolhida": resumo["epoca_escolhida"],
                            "p": resumo["n_params_circuito"],
                            "n_qubits": resumo["n_qubits"],
                        }
                    )
                    if verbose:
                        decorrido = time.perf_counter() - t_inicio
                        restante = decorrido / i * (total - i)
                        print(
                            f"[{i:3d}/{total}] {encoding:12s} {dataset:8s} eta={eta:<5g} "
                            f"seed={seed} val={resumo['acc_val']:.3f} "
                            f"(faltam ~{restante / 60:.1f} min)"
                        )

    por_treino = pd.DataFrame(linhas, columns=list(COLUNAS_SENSIBILIDADE))
    resumo = resumir_sensibilidade(por_treino, protocolo.eta, protocolo.eta_tolerancia_empate)

    metrics = out / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    por_treino.to_csv(metrics / "sensibilidade_eta.csv", index=False)
    resumo.to_csv(metrics / "resumo_sensibilidade_eta.csv", index=False)
    if verbose:
        print("\nlr escolhido pela validação:")
        print(
            resumo.pivot(index="encoding", columns="dataset", values="eta_escolhido").to_string()
        )
        print(f"escrito em {metrics}")
    return por_treino


def escolher_eta(
    por_treino: pd.DataFrame,
    eta_protocolo: float,
    tolerancia: float = PADRAO.eta_tolerancia_empate,
) -> pd.DataFrame:
    """O eta de maior acc_val média nas sementes, por (codificação, dataset).

    Só `acc_val` entra aqui — a coluna `acc_teste` nem é lida. Escolher pelo
    teste transformaria a acurácia de teste reportada numa estimativa
    otimista, que é justamente o que a separação treino/validação/teste evita.

    Empate (diferença absoluta até `tolerancia`, que vem de
    `Protocolo.eta_tolerancia_empate` e só absorve erro de ponto flutuante):
    fica o eta do protocolo, se ele está entre os empatados — a verificação só
    deve "mudar" o eta quando houver ganho de fato na validação; senão, o menor
    eta empatado, que é o passo mais conservador.
    """
    medias = por_treino.groupby(["encoding", "dataset", "eta"])["acc_val"].mean()
    escolhas = []
    for (encoding, dataset), fatia in medias.groupby(level=["encoding", "dataset"]):
        fatia = fatia.droplevel(["encoding", "dataset"])
        melhor = fatia.max()
        empatados = sorted(eta for eta, v in fatia.items() if np.isclose(v, melhor, rtol=0, atol=tolerancia))
        protocolo_empatado = [eta for eta in empatados if np.isclose(eta, eta_protocolo)]
        eta = protocolo_empatado[0] if protocolo_empatado else empatados[0]
        escolhas.append({"encoding": encoding, "dataset": dataset, "eta_escolhido": eta})
    return pd.DataFrame(escolhas)


def resumir_sensibilidade(
    por_treino: pd.DataFrame,
    eta_protocolo: float,
    tolerancia: float = PADRAO.eta_tolerancia_empate,
) -> pd.DataFrame:
    """O eta escolhido pela validação ao lado do eta do protocolo, por (enc, dataset).

    Para cada lado vêm média e desvio de `acc_val` e de `acc_teste` nas
    sementes. Ver os dois lado a lado é o que responde à pergunta: se a
    acurácia de teste no eta escolhido não muda em relação à do protocolo
    além do desvio, a codificação já estava bem treinada com o eta comum.
    """
    escolhas = escolher_eta(por_treino, eta_protocolo, tolerancia)

    def _agregar(fatia: pd.DataFrame, sufixo: str) -> dict:
        return {
            f"acc_val_mean_{sufixo}": fatia["acc_val"].mean(),
            f"acc_val_std_{sufixo}": fatia["acc_val"].std(),
            f"acc_teste_mean_{sufixo}": fatia["acc_teste"].mean(),
            f"acc_teste_std_{sufixo}": fatia["acc_teste"].std(),
        }

    linhas = []
    for escolha in escolhas.itertuples(index=False):
        do_par = por_treino[
            (por_treino["encoding"] == escolha.encoding)
            & (por_treino["dataset"] == escolha.dataset)
        ]
        no_escolhido = do_par[np.isclose(do_par["eta"], escolha.eta_escolhido)]
        no_protocolo = do_par[np.isclose(do_par["eta"], eta_protocolo)]
        linhas.append(
            {
                "encoding": escolha.encoding,
                "dataset": escolha.dataset,
                "eta_escolhido": escolha.eta_escolhido,
                **_agregar(no_escolhido, "escolhido"),
                "eta_protocolo": eta_protocolo,
                **_agregar(no_protocolo, "protocolo"),
                "n_sementes": int(no_escolhido["seed"].nunique()),
                "p": do_par["p"].iloc[0],
                "n_qubits": do_par["n_qubits"].iloc[0],
            }
        )
    return pd.DataFrame(linhas)
