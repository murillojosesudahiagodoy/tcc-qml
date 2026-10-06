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
    # Frações do conjunto TOTAL: 60/20/20 -> 180/60/60 com N = 300. A validação
    # escolhe a época de parada; o teste só é tocado uma vez, no fim.
    val_size: float = 0.2
    test_size: float = 0.2
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

    # --- contabilidade de custo em hardware: S nas contagens (2p + 1)|B|S ---
    shots: int = 1000

    # --- grade de experimentos e varredura ---
    # Acrescentar uma codificação do Bloco C à comparação é acrescentar uma
    # string aqui, desde que ela esteja registrada em ENCODINGS.
    encodings_grade: tuple[str, ...] = ("angle", "amplitude", "reuploading", "zz")
    datasets_grade: tuple[str, ...] = ("xor", "moons", "circles")
    L_reup: int = 3
    L_reup_varredura: tuple[int, ...] = (1, 2, 3, 4, 5)
    # Controle da varredura de L: angle com k camadas tem o mesmo p = 6k do
    # re-uploading com L = k, mas o espectro fixo em {-1, 0, 1}.
    n_layers_varredura: tuple[int, ...] = (1, 2, 3, 4, 5)

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
        `batch_size`. Com N = 300, val_size = 0.2 e test_size = 0.2 são 180
        amostras em 9 lotes de 20 — é o |B| = 20 que entra nas
        contagens (2p + 1)|B| e (2p + 1)|B|S.
        """
        n_treino = round(self.n_samples * (1 - self.val_size - self.test_size))
        n_lotes = max(1, n_treino // self.batch_size)
        return int(np.ceil(n_treino / n_lotes))


PADRAO = Protocolo()
