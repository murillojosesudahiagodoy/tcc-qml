"""Medição do espectro de Fourier do modelo.

O resultado de Fourier (Schuld et al., 2021; Cap. 2, "O resultado geral")
virando número medido. Quando os dados entram linearmente nos ângulos de
rotações de Pauli, a saída é uma série de Fourier truncada nos atributos:

    f(x) = sum_{omega in Omega} c_omega e^{i <omega, x>}

e quem determina o suporte Omega é a CODIFICAÇÃO, não o ansatz; o ansatz e o
observável determinam quais coeficientes c_omega são realizáveis. A tabela de
suportes do Cap. 2 (tab:espectros) dá Omega = {-1, 0, 1} por atributo para o
angle encoding e Omega = {-R, ..., R} para o re-uploading com R repetições.

Omega é um LIMITE SUPERIOR: diz quais frequências podem aparecer, não que
todas apareçam. Duas medições confrontam o limite:

- `espectro()` varre um atributo com o outro fixo (FFT 1D) e mostra o perfil
  de |c_omega| numa direção — é o que a figura desenha;
- `espectro_2d()` avalia f numa grade inteira de [0, 2pi)^2 (FFT 2D) e conta
  os termos (omega_1, omega_2) que de fato têm energia, termos cruzados
  incluídos. É esse número, e não (2 omega_max + 1)^d, que se compara aos
  (2R + 1)^d termos previstos.

Onde NÃO se aplica, e por quê — documentado aqui para ninguém tentar depois:

- **amplitude**: a saída é uma função quadrática racional das amplitudes
  normalizadas, não uma série de Fourier em x. A FFT devolveria um espectro
  denso sem significado teórico.
- **zz**: as frequências vivem nas fases phi_ij(x) = (pi - x_i)(pi - x_j), que
  são produtos de atributos. Varrer x_1 com x_2 fixo dá frequências em x_1, mas
  elas dependem do x_2 escolhido — não são o Omega da tabela de suportes.

Uma assimetria MEDIDA, que vale reportar no Capítulo 4: com dois qubits, o par
de CNOTs em anel do StronglyEntanglingLayers leva Z_0 exatamente em Z_1 na
imagem de Heisenberg. O efeito é que o último bloco S(x) não alcança x_1:
varrendo-o, o espectro medido vai só até R - 1, enquanto em x_2 vai até R.
Com R = 1 o modelo ignora x_1 por completo. O limite teórico é sobre o
que a CODIFICAÇÃO torna acessível; o que se perde num atributo é consequência
da arquitetura (anel de CNOTs + medição de Z_0), não da codificação. Por isso
a tabela reporta o omega de CADA atributo, e não só o máximo entre eles.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pennylane import numpy as pnp

from tccqml.config import PADRAO
from tccqml.model import Classificador, pesos_iniciais

# Codificações para as quais a leitura de espectro é teoricamente válida.
ENCODINGS_COM_ESPECTRO = ("angle", "reuploading")


@dataclass
class Espectro2D:
    """Saída de uma FFT 2D de f sobre [0, 2pi)^2."""

    termos: list[tuple[int, int]]  # (omega_1, omega_2) com energia acima do limiar
    limiar: float

    @property
    def n_termos(self) -> int:
        return len(self.termos)

    @property
    def n_cruzados(self) -> int:
        """Termos que dependem dos dois atributos ao mesmo tempo."""
        return sum(1 for a, b in self.termos if a != 0 and b != 0)

    def omega_max(self, atributo: int) -> int:
        """Maior |omega| alcançado na direção de um atributo."""
        return max((abs(t[atributo]) for t in self.termos), default=0)


@dataclass
class Espectro:
    """Saída de uma medição de espectro ao longo de um atributo."""

    frequencias: np.ndarray  # omega >= 0 com energia acima do limiar
    amplitudes: np.ndarray  # |c_omega| para omega = 0, 1, 2, ...
    omega_max: int  # maior frequência com energia relevante
    encoding: str
    limiar: float

    @property
    def omega(self) -> set[int]:
        """Omega completo, simétrico: f é real, então c_{-w} = conj(c_w)."""
        return {int(w) for w in self.frequencias} | {-int(w) for w in self.frequencias}


def espectro(
    clf: Classificador,
    weights=None,
    alpha=None,
    x2_fixo: float = np.pi / 2,
    indice: int = 0,
    n_pontos: int = PADRAO.espectro_n_pontos,
    limiar: float = PADRAO.espectro_limiar,
) -> Espectro:
    """Fixa theta, varre um atributo em [0, 2pi) e tira a FFT de f(x).

    A varredura vai até 2pi (e não até pi, o intervalo dos dados) porque é o
    período completo: só assim as frequências saem como inteiros exatos e a
    leitura de Omega é limpa. `n_pontos` amostras uniformes num período dão
    frequências inteiras sem vazamento espectral, então o limiar pode ser bem
    baixo — o que sobra acima dele é sinal, não erro numérico.
    """
    if weights is None:
        weights, alpha_padrao, _ = pesos_iniciais(clf)
        alpha = alpha_padrao if alpha is None else alpha

    grade = np.linspace(0.0, 2 * np.pi, n_pontos, endpoint=False)
    X = np.full((n_pontos, _n_features(clf)), float(x2_fixo))
    X[:, indice] = grade

    f = np.asarray(clf.circuit(pnp.array(X, requires_grad=False), weights, alpha), dtype=float)

    coef = np.fft.rfft(f) / n_pontos
    amplitudes = np.abs(coef)
    # O termo omega = 0 não é duplicado; os demais aparecem em +w e -w, então a
    # amplitude física de cada frequência não nula é o dobro do bin da rfft.
    amplitudes[1:] *= 2.0

    relevantes = np.nonzero(amplitudes > limiar)[0]
    return Espectro(
        frequencias=relevantes.astype(int),
        amplitudes=amplitudes,
        omega_max=int(relevantes.max()) if len(relevantes) else 0,
        encoding=clf.encoding,
        limiar=float(limiar),
    )


def espectro_2d(
    clf: Classificador,
    weights=None,
    alpha=None,
    n_pontos: int = PADRAO.espectro_n_pontos_2d,
    limiar: float = PADRAO.espectro_limiar,
) -> Espectro2D:
    """FFT 2D de f numa grade uniforme de [0, 2pi)^2.

    Diferente da varredura 1D, nada fica fixo: todos os termos da série,
    cruzados incluídos, aparecem com o seu coeficiente. `n_pontos` por eixo só
    precisa passar de 2 * omega_max para não haver aliasing; 32 cobre até
    omega = 15, folga de sobra para R <= 5.
    """
    if _n_features(clf) != 2:
        raise ValueError("espectro_2d só está definido para d = 2 atributos")
    if weights is None:
        weights, alpha_padrao, _ = pesos_iniciais(clf)
        alpha = alpha_padrao if alpha is None else alpha

    eixo = np.linspace(0.0, 2 * np.pi, n_pontos, endpoint=False)
    g1, g2 = np.meshgrid(eixo, eixo, indexing="ij")
    X = np.column_stack([g1.ravel(), g2.ravel()])
    f = np.asarray(clf.circuit(pnp.array(X, requires_grad=False), weights, alpha), dtype=float)

    coef = np.abs(np.fft.fft2(f.reshape(n_pontos, n_pontos))) / n_pontos**2
    frequencias = np.fft.fftfreq(n_pontos, d=1.0 / n_pontos).astype(int)
    termos = sorted(
        (int(frequencias[i]), int(frequencias[j])) for i, j in zip(*np.nonzero(coef > limiar))
    )
    return Espectro2D(termos=termos, limiar=float(limiar))


def _n_features(clf: Classificador) -> int:
    """Quantos atributos o circuito espera na entrada.

    Coincide com o número de qubits em todas as codificações do núcleo menos o
    amplitude, onde d = 2^n — mas o amplitude está fora do escopo do espectro.
    """
    if clf.encoding == "amplitude":
        return 2**clf.n_qubits
    return clf.n_qubits


def espectro_por_atributo(
    clf: Classificador,
    weights=None,
    alpha=None,
    x_fixo: float = np.pi / 2,
    n_pontos: int = PADRAO.espectro_n_pontos,
    limiar: float = PADRAO.espectro_limiar,
) -> list[Espectro]:
    """Mede o espectro varrendo cada atributo, um de cada vez."""
    return [
        espectro(
            clf,
            weights,
            alpha,
            x2_fixo=x_fixo,
            indice=i,
            n_pontos=n_pontos,
            limiar=limiar,
        )
        for i in range(_n_features(clf))
    ]


def omega_previsto(encoding: str, R: int = PADRAO.R) -> set[int] | None:
    """Omega previsto por atributo (tab:espectros, Cap. 2). None onde não se aplica."""
    if encoding == "angle":
        return {-1, 0, 1}
    if encoding == "reuploading":
        return set(range(-R, R + 1))
    return None


def n_termos_previsto(encoding: str, d: int, R: int = PADRAO.R) -> int | None:
    """|Omega|^d: 9 para o angle em d = 2; (2R+1)^d para o re-uploading.

    Conta elementos do SUPORTE, não coeficientes ajustáveis. Com d = 2 e R = 3
    são 49 elementos contra p = 18 parâmetros, de modo que os coeficientes não
    podem ser escolhidos de forma independente — é a motivação da hipótese H5
    (Cap. 2, "Número de parâmetros"), não uma demonstração de saturação.
    """
    omega = omega_previsto(encoding, R)
    return None if omega is None else len(omega) ** d


def tabela_espectro(
    n_features: int = 2,
    valores_R: tuple[int, ...] = (1, 2, 3),
    L_var: int = PADRAO.L_var,
    seed: int = PADRAO.seed,
) -> pd.DataFrame:
    """Omega medido contra Omega previsto, uma linha por configuração.

    Inclui o `angle` de propósito: é a referência de menor suporte contra a
    qual o re-uploading é medido (ver `test_angle_nao_tem_energia_acima_da_frequencia_1`).

    Colunas de medição, todas lidas da FFT 2D:

    - `omega_max_x1`, `omega_max_x2`: omega máximo em cada direção;
    - `n_termos_medido`: termos (omega_1, omega_2) com energia;
    - `n_termos_cruzados`: desses, os que dependem dos dois atributos;
    - `dentro_do_limite`: nenhuma energia fora de Omega^d, que é o que a
      teoria garante;
    - `atinge_limite`: os dois atributos chegam ao omega previsto, que é o que
      a teoria NÃO garante e que a arquitetura pode impedir.
    """
    from tccqml import model

    linhas = []
    configuracoes: list[tuple[str, dict | None, int | None]] = [("angle", None, None)]
    configuracoes += [("reuploading", {"R": R}, R) for R in valores_R]

    for enc, kwargs, R in configuracoes:
        clf = model.build(enc, n_features=n_features, L_var=L_var, enc_kwargs=kwargs)
        w, alpha, _ = pesos_iniciais(clf, seed=seed)
        # Pesos aleatórios pequenos podem esconder frequências altas por
        # coeficiente quase nulo; a escala maior excita todo o espectro
        # acessível (mesmo motivo de `_pesos_excitados`, em test_espectro.py).
        w = w * 3.0
        e2 = espectro_2d(clf, w, alpha)
        por_atributo = [e2.omega_max(i) for i in range(n_features)]
        previsto = max(omega_previsto(enc, R or PADRAO.R))
        linhas.append(
            {
                "encoding": enc,
                "R": R,
                "omega_max_previsto": previsto,
                **{f"omega_max_x{i + 1}": v for i, v in enumerate(por_atributo)},
                "n_termos_previsto": n_termos_previsto(enc, n_features, R or PADRAO.R),
                "n_termos_medido": e2.n_termos,
                "n_termos_cruzados": e2.n_cruzados,
                "n_params_circuito": clf.n_params_circuito,
                "dentro_do_limite": max(por_atributo) <= previsto,
                "atinge_limite": min(por_atributo) == previsto,
            }
        )
    return pd.DataFrame(linhas)
