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
    """Os valores congelados da comparação (Cap. 3, tab:protocolo).

    Mudar qualquer campo daqui muda TODOS os experimentos de uma vez, que é o
    ponto: a codificação tem que ser a única coisa que varia.
    """

    # --- dados (Cap. 3, "Conjuntos de dados") ---
    n_samples: int = 300
    noise: float = 0.15
    # Frações do conjunto TOTAL: 60/20/20 -> 180/60/60 com N = 300. A validação
    # escolhe a época de parada; o teste só é tocado uma vez, no fim.
    val_size: float = 0.2
    test_size: float = 0.2
    feature_range: tuple[float, float] = (0.0, np.pi)
    circles_factor: float = 0.4

    # --- ansatz congelado (Cap. 2, "O ansatz variacional"): L_var = L_var ---
    ansatz: str = "strongly_entangling"
    L_var: int = 2

    # --- otimização ---
    epocas: int = 30
    batch_size: int = 20
    eta: float = 0.1
    # Verificação de sensibilidade da taxa de aprendizado (`sensibilidade`),
    # fora da comparação principal, que continua com `eta` para todas. Os
    # candidatos foram fixados ANTES de qualquer execução desta verificação,
    # para a grade não ser ajustada depois de ver resultado, e o orçamento
    # (4 valores x 5 sementes) é o mesmo para todas as codificações: nenhuma
    # ganha mais tentativas que as outras. A escolha entre eles é pela
    # validação; o teste não participa.
    eta_candidatos: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3)
    # Diferença absoluta abaixo da qual duas médias de acc_val empatam na
    # escolha do eta. Só absorve erro de ponto flutuante: com 60 amostras de
    # validação e 5 sementes as médias são múltiplos de 1/300, então médias
    # distintas diferem em pelo menos ~3e-3, muito acima deste limiar.
    eta_tolerancia_empate: float = 1e-9

    # --- reprodutibilidade ---
    seed: int = 42
    sementes: tuple[int, ...] = (42, 43, 44, 45, 46)

    # --- contabilidade de custo em hardware: S nas contagens (2p + 1)|B|S ---
    shots: int = 1000

    # --- grade de experimentos e varredura ---
    # Acrescentar uma codificação à comparação é acrescentar uma
    # string aqui, desde que ela esteja registrada em ENCODINGS.
    encodings_grade: tuple[str, ...] = ("angle", "amplitude", "reuploading", "zz")
    datasets_grade: tuple[str, ...] = ("xor", "moons", "circles")
    # Repetições R do data re-uploading.
    R: int = 3
    R_varredura: tuple[int, ...] = (1, 2, 3, 4, 5)
    # Controle da varredura de R: angle com L_var = k camadas tem o mesmo p = 6k
    # do re-uploading com R = k, mas o espectro fixo em {-1, 0, 1}.
    L_var_varredura: tuple[int, ...] = (1, 2, 3, 4, 5)

    # --- verificações (`verificacoes`): fora da comparação principal ---
    # Hiperparâmetros da regressão logística do controle linear, FIXADOS antes
    # de rodar: não há busca, nem pela validação, para o controle não ganhar
    # um ajuste que o circuito quântico não teve.
    controle_linear_C: float = 1.0
    controle_linear_max_iter: int = 1000

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
