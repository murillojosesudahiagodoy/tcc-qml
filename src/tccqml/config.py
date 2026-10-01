"""Protocolo experimental num lugar só.

Todo parâmetro que precisa ser idêntico entre codificações mora aqui, e não
como valor padrão espalhado pelas funções: é o que impede o código e o texto
de divergirem sem ninguém notar.

Este módulo é a fonte fiel para escrever o Capítulo 3: o que está escrito
aqui é o que de fato roda.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Protocolo:
    """Os valores congelados da comparação (Seção 2.2.3.3 + Etapa 9).

    Mudar qualquer campo daqui muda TODOS os experimentos de uma vez, que é o
    ponto: a codificação tem que ser a única coisa que varia.
    """

    # --- dados (Seção 2.5.7) ---
    n_samples: int = 300
    noise: float = 0.15
    test_size: float = 0.3
    feature_range: tuple[float, float] = (0.0, np.pi)
    circles_factor: float = 0.4

    # --- ansatz congelado (Seção 2.2.3.4) ---
    ansatz: str = "strongly_entangling"
    n_layers: int = 2

    # --- otimização ---
    epocas: int = 30
    batch_size: int = 20
    lr: float = 0.1

    # --- reprodutibilidade ---
    seed: int = 42
    sementes: tuple[int, ...] = (42, 43, 44, 45, 46)

    # --- contabilidade de custo em hardware (Eq. 2.48) ---
    shots: int = 1000

    # --- grade de experimentos e varredura ---
    # Acrescentar uma codificação do Bloco C à comparação é acrescentar uma
    # string aqui, desde que ela esteja registrada em ENCODINGS.
    encodings_grade: tuple[str, ...] = ("angle", "amplitude", "reuploading", "zz")
    datasets_grade: tuple[str, ...] = ("xor", "moons", "circles")
    L_reup: int = 3
    L_reup_varredura: tuple[int, ...] = (1, 2, 3, 4, 5)

    # --- espectro ---
    espectro_n_pontos: int = 512
    espectro_n_pontos_2d: int = 32
    espectro_limiar: float = 1e-6

    # --- saída ---
    out: str = "results"

    @property
    def batch_efetivo(self) -> int:
        """Tamanho médio real dos lotes de treino.

        `treinar()` divide as amostras de treino em `n // batch_size` lotes com
        `np.array_split`, então o lote real é ligeiramente maior que
        `batch_size`. Com N = 300 e test_size = 0.3 são 210 amostras em 10
        lotes de 21 — é o |B| = 21 que aparece na Eq. 2.48.
        """
        n_treino = round(self.n_samples * (1 - self.test_size))
        n_lotes = max(1, n_treino // self.batch_size)
        return int(np.ceil(n_treino / n_lotes))


PADRAO = Protocolo()
