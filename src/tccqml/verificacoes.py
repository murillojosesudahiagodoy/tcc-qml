"""Verificações adicionais pedidas pela orientação — fora da comparação principal.

Duas perguntas sobre os DADOS, não sobre os circuitos:

1. **Controle linear clássico.** O espaço gerado pelas nove funções de
   Fourier de grau 1 por atributo (a base que o angle encoding alcança,
   Omega = {-1, 0, 1} por atributo) contém um separador linear para cada
   conjunto? Se não contém, nenhum classificador que seja combinação linear
   dessas funções separa o conjunto, e uma acurácia baixa do angle deixa de
   ser surpresa.
2. **Limiar de g no circles.** A regra g(x)/2 > t, com g(x) = sin x1 + sin x2
   (Cap. 3, "Previsões e hipóteses"), máxima no centro do quadrado
   normalizado, separa o círculo interno do externo com os raios e o
   reescalonamento reais? Roda sem ruído (a
   geometria pura) e com o ruído do protocolo.

Nada aqui muda dado, partição, normalização ou semente: tudo sai de
`load_dataset` com os padrões de `PADRAO`, exatamente como na grade. O teste
só é usado para a avaliação final — o limiar é escolhido no treino e a
regressão logística tem hiperparâmetros fixados antes de rodar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tccqml.config import PADRAO, Protocolo
from tccqml.data import load_dataset

NOMES_CARACTERISTICAS = (
    "1",
    "cos x1",
    "sin x1",
    "cos x2",
    "sin x2",
    "cos x1 cos x2",
    "cos x1 sin x2",
    "sin x1 cos x2",
    "sin x1 sin x2",
)


def _carregar(dataset: str, seed: int, protocolo: Protocolo, noise: float | None = None):
    """O mesmo `load_dataset` da grade (`experiments.rodar_um`), campo a campo."""
    return load_dataset(
        dataset,
        n_samples=protocolo.n_samples,
        noise=protocolo.noise if noise is None else noise,
        val_size=protocolo.val_size,
        test_size=protocolo.test_size,
        seed=seed,
        feature_range=protocolo.feature_range,
    )


def _gravar(df: pd.DataFrame, resumo: pd.DataFrame, nome: str, out: Path, verbose: bool) -> None:
    metrics = out / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics / f"{nome}.csv", index=False)
    resumo.to_csv(metrics / f"resumo_{nome}.csv", index=False)
    if verbose:
        print(resumo.round(3).to_string(index=False))
        print(f"escrito em {metrics / f'{nome}.csv'} e resumo_{nome}.csv\n")


# --------------------------------------------------------------------------
# Verificação 1 — controle linear nas nove funções de Fourier
# --------------------------------------------------------------------------


def caracteristicas_fourier(X: np.ndarray) -> np.ndarray:
    """As nove funções {1, cos, sin} x {1, cos, sin}, sem o par (1, 1) repetido.

    Calculadas nas coordenadas JÁ normalizadas para [0, pi], as mesmas que
    entram no circuito: no espaço bruto as funções seriam outras. A ordem é a
    de `NOMES_CARACTERISTICAS`.
    """
    X = np.asarray(X, dtype=float)
    c1, s1 = np.cos(X[:, 0]), np.sin(X[:, 0])
    c2, s2 = np.cos(X[:, 1]), np.sin(X[:, 1])
    return np.column_stack(
        [np.ones(len(X)), c1, s1, c2, s2, c1 * c2, c1 * s2, s1 * c2, s1 * s2]
    )


def rodar_controle_linear(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    datasets: tuple[str, ...] | None = None,
    sementes: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Regressão logística sobre as nove funções de Fourier, por dataset e semente.

    Mede se EXISTE um separador linear nesse espaço de funções. Um separador
    encontrado aqui NÃO prova que o circuito quântico consiga realizar esses
    coeficientes, nem que o treino variacional chegue a eles: o modelo
    quântico só alcança as combinações que o ansatz e a medição permitem, sob
    as restrições de norma de um valor esperado. A leitura útil é a negativa:
    se nem a regressão logística separa o conjunto, a limitação é do espaço de
    funções, não da otimização do circuito.

    `C` e `max_iter` vêm de `Protocolo` e foram fixados antes de rodar; não há
    ajuste por validação nem por teste.
    """
    out = Path(out or protocolo.out)
    datasets = datasets or protocolo.datasets_grade
    sementes = sementes or protocolo.sementes

    linhas = []
    for dataset in datasets:
        for seed in sementes:
            ds = _carregar(dataset, seed, protocolo)
            clf = LogisticRegression(
                C=protocolo.controle_linear_C, max_iter=protocolo.controle_linear_max_iter
            ).fit(caracteristicas_fourier(ds.X_train), ds.y_train)
            linhas.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "acc_treino": clf.score(caracteristicas_fourier(ds.X_train), ds.y_train),
                    "acc_val": clf.score(caracteristicas_fourier(ds.X_val), ds.y_val),
                    "acc_teste": clf.score(caracteristicas_fourier(ds.X_test), ds.y_test),
                }
            )

    df = pd.DataFrame(linhas)
    resumo = _media_desvio(df, ["dataset"], ["acc_treino", "acc_val", "acc_teste"])
    if verbose:
        print("controle linear (regressão logística nas 9 funções de Fourier):")
    _gravar(df, resumo, "controle_linear", out, verbose)
    return df


def _media_desvio(df: pd.DataFrame, chaves: list[str], colunas: list[str]) -> pd.DataFrame:
    resumo = df.groupby(chaves, as_index=False, sort=False)[colunas].agg(["mean", "std"])
    resumo.columns = ["_".join(c).rstrip("_") for c in resumo.columns]
    return resumo.assign(n_sementes=df.groupby(chaves, sort=False)["seed"].nunique().values)


# --------------------------------------------------------------------------
# Verificação 2 — limiar de g no circles
# --------------------------------------------------------------------------


def g(X: np.ndarray) -> np.ndarray:
    """g(x) = sin x1 + sin x2 nas coordenadas normalizadas para [0, pi].

    A regra do texto compara g/2 com o limiar t: a divisão por 2 põe g no
    intervalo [-1, 1] da saída do modelo.

    sin é máximo em pi/2, que é aproximadamente onde o reescalonamento põe o
    centro dos círculos: o círculo interno tende a g alto, o externo a g
    baixo. A verificação mede se "tende" vira "separa".
    """
    X = np.asarray(X, dtype=float)
    return np.sin(X[:, 0]) + np.sin(X[:, 1])


def aplicar_limiar(valores: np.ndarray, limiar: float, sentido: str) -> np.ndarray:
    """Classe 1 quando valor > t (sentido ">") ou quando valor < t (sentido "<").

    No circles, `valores` é g/2, a quantidade que o texto compara com t.
    """
    if sentido == ">":
        return (np.asarray(valores) > limiar).astype(int)
    if sentido == "<":
        return (np.asarray(valores) < limiar).astype(int)
    raise ValueError(f"sentido inválido: {sentido!r}")


def escolher_limiar(g_treino: np.ndarray, y_treino: np.ndarray) -> tuple[float, str]:
    """O limiar t e o sentido de maior acurácia NO TREINO.

    Só recebe dados de treino, de propósito: a assinatura não deixa a
    validação nem o teste entrarem na escolha. Os candidatos são os pontos
    médios entre valores consecutivos de g/2 no treino, mais um abaixo do mínimo
    e um acima do máximo, então nenhuma amostra de treino cai em cima do
    limiar.

    Empates: o sentido ">" vem antes (é o que a geometria prevê); entre
    limiares empatados fica o do MEIO da lista de empatados, que tende a ser o
    de maior margem num intervalo contíguo — em vez do primeiro, que encostaria
    numa das classes.
    """
    g = np.asarray(g_treino, dtype=float)
    y = np.asarray(y_treino, dtype=int)
    valores = np.unique(g)
    candidatos = np.concatenate(
        [[valores[0] - 1.0], (valores[:-1] + valores[1:]) / 2, [valores[-1] + 1.0]]
    )
    melhor = None
    for sentido in (">", "<"):
        acertos = np.array([(aplicar_limiar(g, t, sentido) == y).mean() for t in candidatos])
        acc = acertos.max()
        if melhor is None or acc > melhor[0]:
            empatados = candidatos[np.isclose(acertos, acc, rtol=0, atol=1e-12)]
            melhor = (acc, float(empatados[len(empatados) // 2]), sentido)
    return melhor[1], melhor[2]


def _descrever_g(g: np.ndarray, y: np.ndarray) -> dict:
    """Média, desvio, mínimo e máximo de g = sin x1 + sin x2 por classe, e quanto se cruzam.

    `g_sobreposicao` é a fração das amostras que cai no intervalo comum às
    duas classes, [max dos mínimos, min dos máximos]: zero quer dizer que
    existe um limiar perfeito.
    """
    d = {}
    for classe in (0, 1):
        gc = g[y == classe]
        d[f"g_media_c{classe}"] = gc.mean()
        d[f"g_desvio_c{classe}"] = gc.std(ddof=1)
        d[f"g_min_c{classe}"] = gc.min()
        d[f"g_max_c{classe}"] = gc.max()
    inicio = max(d["g_min_c0"], d["g_min_c1"])
    fim = min(d["g_max_c0"], d["g_max_c1"])
    d["g_sobreposicao"] = float(((g >= inicio) & (g <= fim)).mean()) if inicio <= fim else 0.0
    return d


def limiar_circles_um(seed: int, noise: float, protocolo: Protocolo = PADRAO) -> dict:
    """Uma semente de uma versão do circles: limiar no treino, avaliação nos três.

    Cada versão (com e sem ruído) passa pelo `load_dataset` inteiro, então o
    scaler é ajustado no treino DAQUELA versão, como na grade. As medidas de
    sobreposição usam só o treino: são a descrição do que a escolha do limiar
    viu.
    """
    ds = _carregar("circles", seed, protocolo, noise=noise)
    g_tr, g_va, g_te = (g(X) for X in (ds.X_train, ds.X_val, ds.X_test))
    # A regra do texto é g/2 > t: o limiar t vive na escala de g/2.
    limiar, sentido = escolher_limiar(g_tr / 2, ds.y_train)
    return {
        "seed": seed,
        "limiar": limiar,
        "sentido": sentido,
        "acc_treino": (aplicar_limiar(g_tr / 2, limiar, sentido) == ds.y_train).mean(),
        "acc_val": (aplicar_limiar(g_va / 2, limiar, sentido) == ds.y_val).mean(),
        "acc_teste": (aplicar_limiar(g_te / 2, limiar, sentido) == ds.y_test).mean(),
        **_descrever_g(g_tr, ds.y_train),
    }


def rodar_limiar_circles(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    sementes: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """A regra "g/2 > t" no circles, sem ruído e com o ruído do protocolo.

    Sem ruído os dois círculos têm raios fixos (1 e `circles_factor`) e a
    pergunta é só geométrica: o reescalonamento para [0, pi] por atributo
    deixa g constante o bastante em cada círculo para um limiar separar? Com
    ruído mede-se quanto disso sobra nos dados que a grade de fato usa.
    """
    out = Path(out or protocolo.out)
    sementes = sementes or protocolo.sementes

    linhas = []
    for versao, noise in (("sem_ruido", 0.0), ("com_ruido", protocolo.noise)):
        for seed in sementes:
            linhas.append(
                {"versao": versao, "ruido": noise, **limiar_circles_um(seed, noise, protocolo)}
            )

    df = pd.DataFrame(linhas)
    resumo = _media_desvio(
        df, ["versao"], ["limiar", "acc_treino", "acc_val", "acc_teste", "g_sobreposicao"]
    )
    if verbose:
        print("regra g/2 > t no circles, g = sin x1 + sin x2 (t escolhido no treino):")
    _gravar(df, resumo, "limiar_circles", out, verbose)
    return df


def rodar_verificacoes(
    protocolo: Protocolo = PADRAO,
    out: str | Path | None = None,
    sementes: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As duas verificações, na ordem. Ambas são clássicas e levam segundos."""
    controle = rodar_controle_linear(protocolo, out, sementes=sementes, verbose=verbose)
    limiar = rodar_limiar_circles(protocolo, out, sementes=sementes, verbose=verbose)
    return controle, limiar
