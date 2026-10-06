"""Datasets clássicos usados nos experimentos de codificação."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tccqml.config import PADRAO


@dataclass
class Dataset:
    """Dataset já dividido e normalizado, pronto para entrar no circuito."""

    name: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def y_train_pm1(self) -> np.ndarray:
        """Rótulos em {-1, +1}, que é a imagem natural de <Z>."""
        return 2 * self.y_train - 1

    @property
    def y_val_pm1(self) -> np.ndarray:
        """Rótulos de validação em {-1, +1}, no mesmo padrão do treino."""
        return 2 * self.y_val - 1


def make_xor(n_samples: int = PADRAO.n_samples, noise: float = PADRAO.noise, seed: int = 42):
    """XOR: duas classes em quadrantes opostos. Não separável linearmente."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    X = X + rng.normal(0.0, noise, size=X.shape)
    return X, y


_GENERATORS = {
    "xor": lambda n, noise, seed: make_xor(n, noise, seed),
    "moons": lambda n, noise, seed: make_moons(n_samples=n, noise=noise, random_state=seed),
    "circles": lambda n, noise, seed: make_circles(
        n_samples=n, noise=noise, factor=PADRAO.circles_factor, random_state=seed
    ),
}


def load_dataset(
    name: str,
    n_samples: int = PADRAO.n_samples,
    noise: float = PADRAO.noise,
    val_size: float = PADRAO.val_size,
    test_size: float = PADRAO.test_size,
    seed: int = PADRAO.seed,
    feature_range: tuple[float, float] = PADRAO.feature_range,
) -> Dataset:
    """Gera, divide em treino/validação/teste e normaliza um dataset sintético.

    `val_size` e `test_size` são frações do conjunto TOTAL (60/20/20 dá
    180/60/60 com N = 300). A divisão é feita em duas etapas, ambas
    estratificadas e com a mesma semente: primeiro sai o teste, depois a
    validação sai do restante. A validação existe para escolher a época de
    parada sem olhar o teste — se o teste escolhesse a época, a acurácia de
    teste reportada deixaria de ser uma estimativa honesta.

    O scaler é ajustado SÓ no treino e aplicado a treino, validação e teste.
    Ajustar no conjunto completo vaza informação dos conjuntos de avaliação e
    infla a acurácia.

    O intervalo padrão [0, pi] é pensado para angle encoding: RY(theta)
    percorre <Z> de +1 a -1 nesse trecho sem dar a volta, então dois
    valores distintos de entrada nunca caem no mesmo estado.

    Os valores padrão vêm de `config.Protocolo` — o protocolo mora num lugar
    só, e o N = 300 da Seção 2.5.7 é o que de fato roda.
    """
    if name not in _GENERATORS:
        raise ValueError(f"dataset desconhecido: {name!r}. Opções: {sorted(_GENERATORS)}")

    X, y = _GENERATORS[name](n_samples, noise, seed)

    X_resto, X_test, y_resto, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # `val_size` é fração do total; dentro do restante ela vira esta razão.
    X_train, X_val, y_train, y_val = train_test_split(
        X_resto,
        y_resto,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=y_resto,
    )

    scaler = MinMaxScaler(feature_range=feature_range).fit(X_train)

    return Dataset(
        name=name,
        X_train=scaler.transform(X_train),
        X_val=scaler.transform(X_val),
        X_test=scaler.transform(X_test),
        y_train=y_train.astype(int),
        y_val=y_val.astype(int),
        y_test=y_test.astype(int),
    )