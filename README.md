# TCC — Quantum Machine Learning

Código, experimentos e resultados do Trabalho de Conclusão de Curso.

**A pergunta do trabalho:** como a escolha da codificação de dados clássicos em estados
quânticos afeta o desempenho e o custo de um classificador variacional? Quatro
codificações são implementadas e comparadas sob um protocolo idêntico — o ansatz, o
otimizador, os dados e as sementes ficam congelados, e só a codificação muda. A exceção
é o re-uploading, em que as camadas treináveis intercaladas com os dados fazem parte da
própria definição da codificação.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux e macOS
pip install -e ".[dev,notebooks]"
pytest
```

As dependências estão declaradas no [`pyproject.toml`](pyproject.toml) — que é a única
fonte delas — então o `pip install -e .` já traz PennyLane, scikit-learn e o resto.
Os extras `dev` (pytest, ruff) e `notebooks` (JupyterLab) são opcionais.

Instalado, o comando fica disponível como `tccqml`; `python -m tccqml` funciona igual e
é o que os exemplos abaixo usam, por não depender do `PATH`.

## Uso

Tudo pela linha de comando; nenhuma edição de código é necessária para trocar de
codificação ou de conjunto de dados.

```bash
python -m tccqml listar                                          # o que está registrado
python -m tccqml treinar --encoding amplitude --dataset moons    # um treino, com o custo
python -m tccqml treinar --encoding reuploading --dataset moons --L-reup 3
python -m tccqml treinar --encoding angle --dataset xor --ansatz local   # a ablação
```

Cada `treinar` imprime o bloco de custo do circuito junto da acurácia: desempenho e custo
lado a lado é a tese do trabalho.

### Reproduzir os resultados do texto

Na ordem, do zero até as tabelas e figuras prontas para o Overleaf:

```bash
python -m tccqml comparar       # a grade: 4 codificações x 3 datasets x 5 sementes
python -m tccqml varredura      # re-uploading com L = 1..5
python -m tccqml espectro       # Omega medido por FFT contra a Tabela 5
python -m tccqml varredura --parametro n-layers   # controle: angle com 1..5 camadas
python -m tccqml ablacao        # ansatz com e sem CNOTs nos três conjuntos (Previsão 3)
python -m tccqml diagnostico    # amplitude em [0,pi] contra [-1,1]
python -m tccqml tabelas        # results/tables/*.tex
python -m tccqml figuras        # results/figures/*.pdf
```

Na máquina do autor (notebook, CPU, sem GPU), `comparar` soma cerca de 8 minutos de
treino e `varredura` cerca de 15; os demais levam poucos minutos. São tempos de treino,
somados da coluna `segundos` de cada CSV; o tempo de parede é um pouco maior.

## Onde cada número do TCC nasce

Nenhum número do Capítulo 4 é digitado à mão: todos saem de um CSV em `results/metrics/`.

| Artefato | Sai de | Produzido por |
|---|---|---|
| `tab_acuracia.tex` | `resumo.csv` | `comparar` → `tabelas` |
| `tab_custo.tex` | `resumo.csv` | `comparar` → `tabelas` |
| `tab_convergencia.tex` | `resumo.csv` | `comparar` → `tabelas` |
| `tab_espectro.tex` | `espectro.csv` | `espectro` → `tabelas` |
| `tab_ablacao.tex` | `ablacao_entrelacamento.csv` | `ablacao` → `tabelas` |
| `tab_mesmo_p.tex` | `varredura_L.csv` e `varredura_camadas.csv` | `varredura` (as duas) → `tabelas` |
| `tab_qualitativa.tex` | `QUALITATIVA`, em `tabelas.py` | `tabelas` |
| `datasets.pdf` | — | `figuras` |
| `curvas-treinamento.pdf` | `comparacao.csv` | `comparar` → `figuras` |
| `fronteiras-aprendidas.pdf` | `results/weights/` | `comparar` → `figuras` |
| `acuracia-vs-L.pdf` | `varredura_L.csv` | `varredura` → `figuras` |
| `acuracia-vs-custo.pdf` | `resumo.csv` | `comparar` → `figuras` |
| `espectro-reuploading.pdf` | medido na hora | `figuras` |
| `custo-por-codificacao.pdf` | `resumo.csv` | `comparar` → `figuras` |

As seis primeiras tabelas são **derivadas e não devem ser editadas à mão**: se um número
estiver estranho, o lugar de corrigir é o experimento. A `tab_qualitativa` é a exceção —
a coluna de dificuldade de implementação é julgamento, não medição. Mesmo ela não se
edita no `.tex`, que `tabelas` sobrescreve: o texto vive na lista `QUALITATIVA`, em
[`tabelas.py`](src/tccqml/tabelas.py), e é lá que se revisa.

## O protocolo congelado

Vive inteiro em [`src/tccqml/config.py`](src/tccqml/config.py), e é a fonte fiel para
escrever o Capítulo 3:

```
ansatz        StronglyEntanglingLayers, L = 2   (n = 2 -> p = 12)
init          N(0, 0.1²), semente fixa
otimizador    Adam, lr = 0.1, 30 épocas, lotes de 20
custo         quadrático sobre rótulos em {-1, +1}
observável    Z_0
dados         N = 300, ruído 0.15, split 210/90, circles factor = 0.4, normalizado em [0, pi]
sementes      42, 43, 44, 45, 46
```

Exceções deliberadas, todas reportadas como tais:

1. o ansatz `local` (sem CNOTs) existe só para a ablação da Previsão 3;
2. `varredura --parametro n-layers` muda as camadas do ansatz no angle, como controle da
   varredura de L (mesmo `p`, espectro fixo); grava num CSV próprio;
3. o `amplitude` usa 1 qubit em vez de 2, porque a codificação é assim — e por isso `p`
   cai de 12 para 6;
4. o `reuploading` não aplica o ansatz depois dos dados: intercala `L` camadas com os
   blocos de dados, e por isso tem `p = 6L` (18 com `L = 3`).

Toda tabela emite `n_qubits` e `p` para a comparação não ser lida como se fosse de
parâmetros iguais.

## Estrutura

```
src/tccqml/
  config.py         o protocolo experimental, num lugar só
  data.py           datasets, split e normalização
  embeddings.py     as quatro codificações (angle, amplitude, reuploading, zz)
  ansatz.py         o circuito treinável — congelado, mais a variante da ablação
  model.py          monta o QNode e conta parâmetros
  train.py          o laço de treinamento
  metrics.py        custo quadrático e acurácia
  circuit_stats.py  qubits, profundidade, portas e custo de gradiente
  espectro.py       medição do espectro de Fourier por FFT (1D e 2D)
  experiments.py    os runners: grade, varreduras, ablação, diagnóstico
  tabelas.py        gera .csv e .tex
  figuras.py        gera .pdf vetorial
  cli.py            a linha de comando
tests/              119 testes
notebooks/          exploração visual de uma combinação de cada vez
results/metrics/    CSVs (não versionados)
results/tables/     .tex para o Overleaf
results/figures/    .pdf para o Overleaf
docs/               reservado para documentação (o conteúdo atual é local,
                    inclusive o texto do TCC em .pdf)
```

## Limitações do estudo

Registradas aqui porque pertencem à seção de limitações do texto:

- **Simulador sem ruído.** Tudo roda em `default.qubit`. O custo em hardware real é
  contabilizado analiticamente (Eqs. 2.47 e 2.48), não medido — inclusive o
  *parameter-shift*, que não chega a ser executado: o treino usa retropropagação.
- **Apenas `d = 2` atributos**, o que preserva a validade das contagens de custo e da
  análise geométrica das fronteiras, mas não diz nada sobre dimensões maiores.
- **Uma única arquitetura de ansatz.** Congelar é o que permite atribuir a diferença à
  codificação, mas pode prejudicar alguma codificação que funcionaria melhor com outro.
- **`p` desigual entre codificações**: o amplitude tem 6 parâmetros contra 12 das demais,
  e o re-uploading com `L = 3` tem 18.
- **Cinco sementes.** Diferença de acurácia menor que o desvio reportado não sustenta
  afirmação.
- **Basis encoding e embeddings treináveis ficam como teoria, sem experimento.** É escolha
  de escopo declarada, não omissão: a Etapa 9 do roteiro não os inclui na lista do que
  implementar, e a Seção 2.4.7 já justifica o caso dos treináveis (o custo de preparação
  passaria a depender do resultado do treino, o que quebra a comparabilidade).

## Convenção de commits

`tipo: descrição curta no imperativo`

- `feat:` nova funcionalidade / experimento
- `fix:` correção
- `refactor:` reorganização sem mudança de comportamento
- `exp:` rodada de experimento
- `docs:` texto do TCC, anotações
- `test:` testes
- `chore:` infraestrutura
