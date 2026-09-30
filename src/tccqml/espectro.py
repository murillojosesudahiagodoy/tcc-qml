"""Medição do espectro de Fourier do modelo.

Este é o resultado teórico central do trabalho virando número medido. Um
classificador variacional cujo bloco de dados é feito de rotações Pauli é uma
série de Fourier truncada nos atributos (Seção 2.4.5):

    f(x) = sum_{omega in Omega} c_omega e^{i <omega, x>}

e quem determina o conjunto de frequências acessíveis Omega é a CODIFICAÇÃO,
não o ansatz. O ansatz só escolhe os coeficientes c_omega. A Tabela 5 (p. 52)
afirma Omega = {-1, 0, 1} por atributo para o angle encoding e
Omega = {-L, ..., L} para o re-uploading com L blocos; `espectro()` confirma ou
refuta isso por FFT.

Onde NÃO se aplica, e por quê — documentado aqui para ninguém tentar depois:

- **amplitude**: a saída é uma função quadrática racional das amplitudes
  normalizadas, não uma série de Fourier em x. A FFT devolveria um espectro
  denso sem significado teórico.
- **zz**: as frequências vivem nas fases phi_ij(x) = (pi - x_i)(pi - x_j), que
  são produtos de atributos. Varrer x_1 com x_2 fixo dá frequências em x_1, mas
  elas dependem do x_2 escolhido — não são o Omega da Tabela 5.

Uma assimetria MEDIDA, que vale reportar no Capítulo 4: com dois qubits, o par
de CNOTs em anel do StronglyEntanglingLayers leva Z_0 exatamente em Z_1 na
imagem de Heisenberg. O efeito é que o último bloco S(x) não alcança o atributo
0: varrendo x_0 o espectro medido vai só até L - 1, enquanto varrendo x_1 vai
até L, como a Tabela 5 prevê. O limite teórico é sobre o que a CODIFICAÇÃO
torna acessível; o que se perde num atributo é consequência do ansatz
congelado, não da codificação. Por isso a medição percorre TODOS os atributos e
reporta o máximo — e guarda os valores por atributo, para a assimetria ficar
visível em vez de escondida.
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


def omega_previsto(encoding: str, L_reup: int = PADRAO.L_reup) -> set[int] | None:
    """Omega que a Tabela 5 (p. 52) prevê por atributo. None onde não se aplica."""
    if encoding == "angle":
        return {-1, 0, 1}
    if encoding == "reuploading":
        return set(range(-L_reup, L_reup + 1))
    return None


def n_termos_previsto(encoding: str, d: int, L_reup: int = PADRAO.L_reup) -> int | None:
    """|Omega|^d: 9 para o angle em d = 2; (2L+1)^d para o re-uploading.

    Com d = 2 e L = 3 isso dá 49 termos de Fourier contra p = 18 parâmetros
    treináveis (ou os 12 da configuração de referência das outras
    codificações). Os coeficientes não podem, portanto, ser escolhidos de forma
    independente — é a explicação teórica da saturação da Previsão 5
    (Seção 2.5.6.1).
    """
    omega = omega_previsto(encoding, L_reup)
    return None if omega is None else len(omega) ** d


def tabela_espectro(
    n_features: int = 2,
    L_reups: tuple[int, ...] = (1, 2, 3),
    n_layers: int = PADRAO.n_layers,
    seed: int = PADRAO.seed,
) -> pd.DataFrame:
    """Omega medido por FFT contra Omega previsto, uma linha por configuração.

    Inclui o `angle` de propósito: ele é o piso contra o qual o re-uploading
    é medido (ver `test_angle_nao_tem_energia_acima_da_frequencia_1`).
    """
    from tccqml import model

    linhas = []
    configuracoes: list[tuple[str, dict | None, int | None]] = [("angle", None, None)]
    configuracoes += [("reuploading", {"L_reup": L}, L) for L in L_reups]

    for enc, kwargs, L in configuracoes:
        clf = model.build(enc, n_features=n_features, n_layers=n_layers, enc_kwargs=kwargs)
        w, alpha, _ = pesos_iniciais(clf, seed=seed)
        # Pesos aleatórios pequenos podem esconder frequências altas por
        # coeficiente quase nulo; a escala maior excita todo o espectro
        # acessível (mesmo motivo de `_pesos_excitados`, em test_espectro.py).
        w = w * 3.0
        espectros = espectro_por_atributo(clf, w, alpha)
        por_atributo = [e.omega_max for e in espectros]
        omega_max = max(por_atributo)
        previsto = omega_previsto(enc, L or PADRAO.L_reup)
        # |Omega| = 2 * omega_max + 1: omega = 0 é sempre acessível (o viés e o
        # termo constante da série), mesmo quando um jogo de pesos específico
        # zera o seu coeficiente.
        n_termos_medido = (2 * omega_max + 1) ** n_features
        linhas.append(
            {
                "encoding": enc,
                "L_reup": L,
                "omega_max_medido": omega_max,
                "omega_max_previsto": max(previsto),
                "omega_max_por_atributo": ";".join(str(v) for v in por_atributo),
                "n_termos_medido": n_termos_medido,
                "n_termos_previsto": n_termos_previsto(enc, n_features, L or PADRAO.L_reup),
                "n_params_circuito": clf.n_params_circuito,
                "confere": omega_max == max(previsto),
            }
        )
    return pd.DataFrame(linhas)
