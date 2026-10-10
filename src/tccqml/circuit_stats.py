"""Contagem de recursos dos circuitos.

Para cada codificação registram-se: número de qubits,
profundidade da codificação, profundidade total, número de portas, número de
portas de dois qubits e número de parâmetros treináveis. É exatamente o que
`stats()` devolve.

Duas decisões que mudam os números e por isso ficam explícitas:

1. **A contagem é feita no circuito já decomposto** nas portas básicas
   (Cap. 2, "Modelo de recursos"): rotações de um eixo, Hadamard e CNOT.
   Sem isso, "prepare o estado |psi(x)>" contaria como uma
   porta só e a comparação perderia o sentido: o amplitude encoding pareceria
   mais barato que o angle. `GlobalPhase` e `Identity` não são contados —
   fase global não é observável e não custa nada em hardware.

2. **As métricas de gradiente são analíticas, não medidas.** O treino roda em
   `default.qubit` com retropropagação, que não paga o custo do
   parameter-shift (Cap. 2, "regra de deslocamento de parâmetro"). As
   contagens descrevem o que o MESMO
   treino custaria em hardware real, e é isso que o TCC reporta: 2p + 1
   avaliações por amostra (as 2p deslocadas do parameter-shift mais a do resíduo
   f(x) + b - y), (2p + 1)|B| avaliações e (2p + 1)|B|S shots por passo.
"""

from collections import Counter

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from tccqml.embeddings import get_encoding
from tccqml.model import Classificador, pesos_iniciais

# Conjunto de portas básicas em que tudo é decomposto antes de contar.
PORTAS_BASICAS = {
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "S",
    "T",
    "CNOT",
    "CZ",
    "GlobalPhase",
    "Identity",
}

# Não contam como porta: a fase global não é observável e a identidade não faz
# nada. Contá-las inflaria a profundidade da codificação sem custo real.
IGNORADAS = {"GlobalPhase", "Identity"}


def _decompor(tape):
    """Reduz a tape às portas básicas (Cap. 2, "Modelo de recursos")."""
    (decomposta,), _ = qml.transforms.decompose(tape, gate_set=PORTAS_BASICAS)
    return decomposta


def contar_tape(tape) -> dict:
    """Contagens de uma tape já construída: portas, profundidade, qubits."""
    tape = _decompor(tape)
    operacoes = [op for op in tape.operations if op.name not in IGNORADAS]
    gates_1q = sum(1 for op in operacoes if len(op.wires) == 1)
    gates_2q = sum(1 for op in operacoes if len(op.wires) >= 2)
    return {
        "n_qubits": len(tape.wires),
        "depth": _profundidade(operacoes),
        "gates_1q": gates_1q,
        "gates_2q": gates_2q,
        "gates_total": gates_1q + gates_2q,
        "gate_types": dict(Counter(op.name for op in operacoes)),
    }


def _profundidade(operacoes) -> int:
    """Profundidade = maior número de portas empilhadas num mesmo qubit.

    Calculada à mão sobre a lista já filtrada, para que `GlobalPhase` (que o
    PennyLane aplica a todos os fios) não some profundidade artificial.
    """
    por_fio: dict = {}
    for op in operacoes:
        fios = list(op.wires)
        nivel = max((por_fio.get(f, 0) for f in fios), default=0) + 1
        for f in fios:
            por_fio[f] = nivel
    return max(por_fio.values(), default=0)


def _tape_do_bloco_de_dados(clf: Classificador, x_exemplo) -> object:
    """QNode auxiliar com APENAS a codificação, para medir sua profundidade.

    No re-uploading isso são os `R` blocos S(x) sem as camadas treináveis
    — ou seja, a soma das profundidades dos blocos de dados, reportada separada
    da profundidade total (Cap. 3, "Métricas e critério de análise").
    """
    enc = get_encoding(clf.encoding, **(clf.enc_kwargs or {}))
    bloco = enc.bloco_de_dados()
    dev = qml.device("default.qubit", wires=clf.n_qubits)

    @qml.qnode(dev)
    def apenas_codificacao(x):
        bloco(x, wires=range(clf.n_qubits))
        return qml.expval(qml.PauliZ(0))

    return qml.workflow.construct_tape(apenas_codificacao, level="device")(x_exemplo)


def aval_derivadas_amostra(n_params_circuito: int) -> int:
    """Só as 2p avaliações DESLOCADAS do parameter-shift, por amostra.

    Analítica. Dá as derivadas parciais de f(x) e mais nada; fica nos CSVs
    para comparação com o texto. O gradiente do custo também precisa do
    resíduo, que é outra avaliação: ver `aval_gradiente_amostra`. O viés
    clássico não entra: sua derivada não executa o circuito.
    """
    return 2 * int(n_params_circuito)


def aval_gradiente_amostra(n_params_circuito: int) -> int:
    """2p + 1 avaliações por amostra para o gradiente do custo quadrático.

    O gradiente de (f(x) + b - y)^2 é 2 (f(x) + b - y) df/dtheta: as 2p
    avaliações deslocadas dão df/dtheta, mas o resíduo exige f(x) no ponto
    sem deslocamento — uma avaliação a mais por amostra. É a contagem
    (2p + 1)|B| que o texto usa (Cap. 2, "regra de deslocamento de parâmetro").
    """
    return aval_derivadas_amostra(n_params_circuito) + 1


def n_aval_passo(n_params_circuito: int, batch_size: int) -> int:
    """Avaliações distintas do circuito por passo do otimizador: (2p + 1)|B|.

    É a contagem de circuitos diferentes a submeter, sem os shots: separar as
    duas coisas deixa claro quanto do custo vem do gradiente e quanto da
    estimativa estatística de cada valor esperado.
    """
    return aval_gradiente_amostra(n_params_circuito) * int(batch_size)


def n_shots_passo(n_params_circuito: int, batch_size: int, shots: int) -> int:
    """Shots por passo do otimizador: (2p + 1)|B|S, com S = `Protocolo.shots`.

    Analítica. Com p = 12, |B| = 20 (`PADRAO.batch_efetivo`) e S = 1000 dá
    500 avaliações x 1000 shots = 500 000 execuções para UM passo, já com a
    avaliação do resíduo.
    """
    return n_aval_passo(n_params_circuito, batch_size) * int(shots)


def stats(
    clf: Classificador,
    x_exemplo,
    weights=None,
    alpha=None,
    batch_size: int | None = None,
    shots: int | None = None,
) -> dict:
    """Métricas de custo estático do circuito de `clf`.

    `x_exemplo` é uma amostra ÚNICA (vetor de `n_features`), não um lote: a
    estrutura do circuito não depende do valor dos dados, só da forma.
    `batch_size` e `shots` têm como padrão o |B| e o S do protocolo.
    """
    from tccqml.config import PADRAO

    batch_size = PADRAO.batch_efetivo if batch_size is None else batch_size
    shots = PADRAO.shots if shots is None else shots
    if weights is None:
        weights, alpha_padrao, _ = pesos_iniciais(clf)
        alpha = alpha_padrao if alpha is None else alpha

    x_exemplo = pnp.array(np.asarray(x_exemplo, dtype=float), requires_grad=False)

    tape_total = qml.workflow.construct_tape(clf.circuit, level="device")(
        x_exemplo, weights, alpha
    )
    total = contar_tape(tape_total)
    codificacao = contar_tape(_tape_do_bloco_de_dados(clf, x_exemplo))

    p = clf.n_params_circuito
    return {
        "encoding": clf.encoding,
        "ansatz": clf.ansatz,
        "n_qubits": total["n_qubits"],
        "depth": total["depth"],
        "depth_encoding": codificacao["depth"],
        "gates_1q": total["gates_1q"],
        "gates_2q": total["gates_2q"],
        "gates_total": total["gates_total"],
        "gates_1q_encoding": codificacao["gates_1q"],
        "gates_2q_encoding": codificacao["gates_2q"],
        "gates_total_encoding": codificacao["gates_total"],
        "n_params_ansatz": clf.n_params_ansatz,
        "n_params_encoding": clf.n_params_encoding,
        "n_params_circuito": p,
        "n_params_modelo": clf.n_params,
        "aval_derivadas_amostra": aval_derivadas_amostra(p),
        "aval_gradiente_amostra": aval_gradiente_amostra(p),
        "n_aval_passo": n_aval_passo(p, batch_size),
        "n_shots_passo": n_shots_passo(p, batch_size, shots),
    }


def stats_ansatz(n_qubits: int, L_var: int, ansatz: str = "strongly_entangling") -> dict:
    """Recursos do ansatz sozinho, sem codificação.

    Separado de `stats()` porque compara ansätze, sem codificação:
    para n qubits e L_var camadas o StronglyEntanglingLayers tem 3 L_var n parâmetros,
    3Ln portas de um qubit, Ln de dois qubits e 6Ln avaliações deslocadas
    (as 2p do parameter-shift; com a do resíduo, 6Ln + 1 por amostra).
    """
    from tccqml.ansatz import get_ansatz

    ans = get_ansatz(ansatz)
    forma = tuple(ans.weights_shape(L_var, n_qubits))
    pesos = pnp.zeros(forma, requires_grad=True)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def apenas_ansatz(w):
        ans.apply(w, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    contagem = contar_tape(qml.workflow.construct_tape(apenas_ansatz, level="device")(pesos))
    p = int(np.prod(forma))
    contagem["n_params_circuito"] = p
    contagem["aval_derivadas_amostra"] = aval_derivadas_amostra(p)
    contagem["aval_gradiente_amostra"] = aval_gradiente_amostra(p)
    return contagem


def formatar(s: dict) -> str:
    """Bloco de texto para a CLI imprimir custo ao lado da acurácia."""
    linhas = [
        f"  qubits ............... {s['n_qubits']}",
        f"  profundidade total ... {s['depth']}",
        f"  profundidade codif. .. {s['depth_encoding']}",
        f"  portas de 1 qubit .... {s['gates_1q']}",
        f"  portas de 2 qubits ... {s['gates_2q']}",
        f"  parâmetros (p) ....... {s['n_params_circuito']}",
        f"  aval. deslocadas ..... {s['aval_derivadas_amostra']}  por amostra (2p, parameter-shift)",
        f"  aval. p/ gradiente ... {s['aval_gradiente_amostra']}  por amostra (2p + 1)",
        f"  aval. por passo ...... {s['n_aval_passo']:,}  ((2p + 1)|B|)",
        f"  shots por passo ...... {s['n_shots_passo']:,}  ((2p + 1)|B|S)",
    ]
    return "\n".join(linhas)
