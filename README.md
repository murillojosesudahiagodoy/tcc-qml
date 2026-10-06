# TCC — Quantum Machine Learning

Código, experimentos e resultados do Trabalho de Conclusão de Curso.

**A pergunta do trabalho:** como a escolha da codificação de dados clássicos em estados
quânticos afeta o desempenho e o custo de um classificador variacional? Quatro
codificações são implementadas e comparadas sob um protocolo idêntico — o ansatz, o
otimizador, os dados e as sementes ficam congelados, e só a codificação muda. A exceção
é o re-uploading, em que as camadas treináveis intercaladas com os dados fazem parte da
própria definição da codificação.

## Setup

Instalação reprodutível, com as versões exatas usadas nos resultados do texto:

```bash
python -m venv .venv && source .venv/bin/activate   # Linux e macOS
# no Windows: python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-lock.txt
pip install -e .
pytest
```

O [`requirements-lock.txt`](requirements-lock.txt) fixa (`==`) todas as dependências do
ambiente em que os resultados foram gerados, inclusive pytest e JupyterLab; o
`pip install -e .` depois dele só instala o próprio pacote, sem trocar nenhuma versão.
O lock foi gerado e testado (`pytest`) no Windows 10, com Python 3.14.3. Em Linux e macOS
ele não foi instalado de fato: só se conferiu que todas as versões fixadas têm wheel para
CPython 3.14 em Linux x86_64 e macOS arm64. O `pywinpty`, que só existe no Windows, leva o
marcador `sys_platform == "win32"`.

As dependências em si estão declaradas, com versões mínimas, no
[`pyproject.toml`](pyproject.toml). Para desenvolver sem fixar versões,
`pip install -e ".[dev,notebooks]"` basta; os extras `dev` (pytest, ruff) e `notebooks`
(JupyterLab, ipywidgets e pymupdf, para o [notebook](#o-notebook)) são opcionais.

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
python -m tccqml sensibilidade  # verificação do lr: 4 candidatos x 5 sementes, escolha pela validação
python -m tccqml varredura      # re-uploading com L = 1..5
python -m tccqml espectro       # Omega medido por FFT contra a Tabela 5
python -m tccqml varredura --parametro n-layers   # controle: angle com 1..5 camadas
python -m tccqml ablacao        # ansatz com e sem CNOTs nos três conjuntos (Previsão 3)
python -m tccqml diagnostico    # amplitude em [0,pi] contra [-1,1]
python -m tccqml verificacoes   # controle linear nas 9 funções de Fourier e limiar de g no circles
python -m tccqml tabelas        # results/tables/*.tex
python -m tccqml figuras        # results/figures/*.pdf
```

Na máquina do autor (notebook, CPU, sem GPU), `comparar` soma cerca de 8 minutos de
treino e `varredura` cerca de 15; os demais levam poucos minutos. São tempos de treino,
somados da coluna `segundos` de cada CSV; o tempo de parede é um pouco maior.

## O notebook

[`notebooks/tcc_qml.ipynb`](notebooks/tcc_qml.ipynb) percorre o mesmo fluxo de forma
didática, de cima a baixo. Ele não reimplementa treino nem análise: cada experimento é uma
chamada a `python -m tccqml <subcomando>` com o mesmo Python do kernel, e cada célula imprime
o comando equivalente. Por isso os números são os mesmos do terminal. As figuras exibidas são
os PDFs gravados por `figuras`. O notebook só desenha o que a CLI não gera: o circuito, o
estado ψ(x), a comparação bruto × normalizado e os gráficos do treino interativo.

| Seção | O que tem |
|---|---|
| 0. Setup | parâmetros, versões do Python e do PennyLane, `pytest` e `listar` |
| 1. Os dados | `datasets.pdf`, hipóteses sobre a fronteira de cada conjunto, bruto × normalizado, tamanhos 180/60/60, fração de classe e min/max por partição, e por que [0, π] |
| 2. Anatomia do classificador | seletor de codificação que redesenha o circuito (em blocos e `level="device"`), o bloco de custo com a contagem (2p + 1) e ψ(x) com amplitudes, probabilidades e concorrência; texto sobre cada desenho, estado produto × entrelaçado, retropropagação × *parameter-shift* e o espectro Ω previsto |
| 3. Treino interativo | painel com as opções de `treinar`, que roda `treinar ... --salvar` em `results/notebook/` e mostra circuito, custo, curvas com a época escolhida, fronteira de decisão, acurácias e a comparação com a média ± desvio da grade; explica a divisão 60/20/20 |
| 4. Os experimentos | uma seção por subcomando (`comparar`, `sensibilidade`, as duas `varredura`, `espectro`, `ablacao`, `diagnostico`, `verificacoes`), com a pergunta, a execução, os CSVs, a tabela de `tabelas`, as figuras de `figuras` e como interpretar |
| 5. Tabelas e figuras finais | todos os `tab_*` e PDFs, com a parte do TCC a que cada um corresponde |
| 6. Resumo | subcomando, pergunta, arquivos, figuras e lugar no TCC; a sequência completa de comandos |

**Como abrir.** Instale o extra `notebooks` (`pip install -e ".[notebooks]"`; o
`requirements-lock.txt` ainda não fixa o ipywidgets) e abra o arquivo no **VS Code**, com o
kernel da `.venv`, ou no **Jupyter** (`jupyter lab notebooks/tcc_qml.ipynb`). O notebook
acha a raiz do repositório sozinho, em Windows ou Linux.

Os painéis das seções 2 e 3 usam ipywidgets, que **só funcionam no Jupyter (Lab ou Notebook)
ou no VS Code**. Em outro ambiente, como a pré-visualização do GitHub ou uma execução por
`nbconvert`, ponha `USAR_WIDGETS = False` na célula de parâmetros: as células alternativas
logo abaixo de cada painel fazem o mesmo a partir de variáveis.

A célula de parâmetros, no topo, controla o resto:

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `OUT_DIR` | `"results"` | o `--out` dos subcomandos e de onde se leem os resultados |
| `RODAR` | `True` | `False` não executa nenhum experimento, só carrega os arquivos |
| `RODAR_TESTES` | `True` | roda o `pytest` no setup (~1 min) |
| `USAR_WIDGETS` | `True` | `False` usa as células alternativas sem ipywidgets |
| `OUT_TREINO` | `"results/notebook"` | saída do treino interativo, separada da oficial |

Cada experimento só roda se faltar algum arquivo que ele grava. Com os resultados já em
`results/`, o notebook inteiro roda em cerca de um minuto. Do zero, com `RODAR = True`,
leva cerca de 50 minutos.

## Onde cada número do TCC nasce

Nenhum número do Capítulo 4 é digitado à mão: todos saem de um CSV em `results/metrics/`.

| Artefato | Sai de | Produzido por |
|---|---|---|
| `tab_acuracia.tex` | `resumo.csv` | `comparar` → `tabelas` |
| `tab_custo.tex` | `resumo.csv` | `comparar` → `tabelas` |
| `tab_convergencia.tex` | `resumo.csv` | `comparar` → `tabelas` |
| `tab_espectro.tex` | `espectro.csv` | `espectro` → `tabelas` |
| `tab_ablacao.tex` | `ablacao_entrelacamento.csv` | `ablacao` → `tabelas` |
| `tab_diagnostico.tex` | `diagnostico_amplitude.csv` | `diagnostico` → `tabelas` |
| `tab_mesmo_p.tex` | `varredura_L.csv` e `varredura_camadas.csv` | `varredura` (as duas) → `tabelas` |
| `tab_sensibilidade.tex` | `resumo_sensibilidade_lr.csv` | `sensibilidade` → `tabelas` |
| `tab_verificacoes.tex` | `controle_linear.csv` e `limiar_circles.csv` | `verificacoes` → `tabelas` |
| `tab_qualitativa.tex` | `QUALITATIVA`, em `tabelas.py` | `tabelas` |
| `datasets.pdf` | — | `figuras` |
| `curvas-treinamento.pdf` | `comparacao.csv` | `comparar` → `figuras` |
| `fronteiras-aprendidas.pdf` | `results/weights/` | `comparar` → `figuras` |
| `acuracia-vs-L.pdf` | `varredura_L.csv` | `varredura` → `figuras` |
| `acuracia-vs-custo.pdf` | `resumo.csv` | `comparar` → `figuras` |
| `espectro-reuploading.pdf` | medido na hora | `figuras` |
| `custo-por-codificacao.pdf` | `resumo.csv` | `comparar` → `figuras` |
| `sensibilidade-lr.pdf` | `sensibilidade_lr.csv` | `sensibilidade` → `figuras` |
| `limiar-circles.pdf` | calculado na hora (semente 42) | `figuras` |

As nove primeiras tabelas são **derivadas e não devem ser editadas à mão**: se um número
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
dados         N = 300, ruído 0.15, circles factor = 0.4, normalizado em [0, pi]
split         treino/validação/teste 180/60/60 (60/20/20), estratificado, semente = a do treino
parada        época de maior acurácia de validação (empate: a mais antiga)
sementes      42, 43, 44, 45, 46
```

A validação escolhe a época de parada: o treino registra custo, acurácia de treino e
acurácia de validação a cada época, guarda os parâmetros da melhor época de validação e os
restaura no fim. O teste é usado **uma única vez**, nesse modelo — nenhuma curva por época
o consulta, para que a acurácia de teste reportada continue sendo uma estimativa honesta.
O `MinMaxScaler` é ajustado só no treino e aplicado aos três conjuntos. Com 180 amostras
de treino, os lotes efetivos têm |B| = 20. Em hardware, o gradiente do custo quadrático
pede 2p + 1 avaliações por amostra (as 2p deslocadas do *parameter-shift* mais a do
resíduo f(x) + b − y), ou seja, (2p + 1)|B| avaliações e (2p + 1)|B|S shots por passo,
com S = 1000: no angle, (2 · 12 + 1) · 20 = 500 avaliações e 500 000 shots.

Exceções deliberadas, todas reportadas como tais:

1. o ansatz `local` (sem CNOTs) existe só para a ablação da Previsão 3;
2. `varredura --parametro n-layers` muda as camadas do ansatz no angle, como controle da
   varredura de L (mesmo `p`, espectro fixo); grava num CSV próprio;
3. o `amplitude` usa 1 qubit em vez de 2, porque a codificação é assim — e por isso `p`
   cai de 12 para 6;
4. o `reuploading` não aplica o ansatz depois dos dados: intercala `L` camadas com os
   blocos de dados, e por isso tem `p = 6L` (18 com `L = 3`).
5. `sensibilidade` repete a grade variando só o lr sobre `lr_candidatos` (0.01, 0.03, 0.1,
   0.3, fixados antes de rodar; 4 valores x 5 sementes para todas as codificações), escolhe o
   lr pela maior acurácia média de validação, sem olhar o teste, e o reporta ao lado do
   lr = 0.1 do protocolo: separa falha de otimização de limitação de representação.

Toda tabela emite `n_qubits` e `p` para a comparação não ser lida como se fosse de
parâmetros iguais.

### Verificações sobre os dados (`verificacoes`)

Duas checagens clássicas, sem circuito, com as mesmas partições, normalização e sementes da
grade; o teste só entra na avaliação final. Gravam em CSVs próprios.

- **Controle linear.** Regressão logística (C = 1, `max_iter` = 1000, fixados antes de rodar)
  sobre as nove funções {1, cos x_j, sin x_j} e seus produtos, nas coordenadas em [0, pi].
  Diz se esse espaço contém um separador — não que o circuito consiga realizá-lo ou aprendê-lo.
- **Limiar de g no circles.** g(x) = (sin x1 + sin x2)/2, com limiar e sentido escolhidos no
  treino e avaliados em validação e teste, sem ruído e com o ruído do protocolo. Registra também
  média, desvio, mínimo e máximo de g por classe, para medir a sobreposição.

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
  verificacoes.py   controle linear clássico e limiar de g no circles
  tabelas.py        gera .csv e .tex
  figuras.py        gera .pdf vetorial
  cli.py            a linha de comando
tests/              177 testes
notebooks/          tcc_qml.ipynb: o fluxo inteiro da CLI, de forma didática
results/metrics/    CSVs de cada experimento
results/tables/     .tex (e .csv) para o Overleaf
results/figures/    .pdf para o Overleaf
results/weights/    pesos finais da grade (.npz), de onde saem as fronteiras
                    (as quatro pastas são versionadas; results/notebook/, não)
docs/               reservado para documentação (o conteúdo atual é local,
                    inclusive o texto do TCC em .pdf)
```

## Limitações do estudo

Registradas aqui porque pertencem à seção de limitações do texto:

- **Simulador sem ruído.** Tudo roda em `default.qubit`. O custo em hardware real é
  contabilizado analiticamente ((2p + 1)|B|S shots por passo), não medido — inclusive o
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

## Licença

Código distribuído sob a licença MIT; o texto completo está em [`LICENSE`](LICENSE).
