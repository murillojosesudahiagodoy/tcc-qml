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
ambiente em que os resultados foram gerados, inclusive pytest, JupyterLab e ipywidgets; o
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
python -m tccqml treinar --encoding reuploading --dataset moons --R 3
python -m tccqml treinar --encoding angle --dataset xor --ansatz local   # a ablação
```

Cada `treinar` imprime o bloco de custo do circuito junto da acurácia: desempenho e custo
lado a lado é a tese do trabalho.

### Reproduzir os resultados do texto

Na ordem, do zero até as tabelas e figuras prontas para o Overleaf:

```bash
python -m tccqml comparar       # a grade: 4 codificações x 3 datasets x 5 sementes
python -m tccqml sensibilidade  # verificação do eta: 4 candidatos x 5 sementes, escolha pela validação
python -m tccqml varredura      # re-uploading com R = 1..5 (--parametro R, o padrão)
python -m tccqml espectro       # Omega medido por FFT contra o suporte previsto (tab:espectros)
python -m tccqml varredura --parametro L_var   # controle: angle com L_var = 1..5
python -m tccqml ablacao        # ansatz com e sem CNOTs nos três conjuntos (hipótese H3)
python -m tccqml diagnostico    # amplitude em [0,pi] contra [-1,1]
python -m tccqml verificacoes   # controle linear nas 9 funções de Fourier e limiar de g no circles
python -m tccqml tabelas        # results/tables/*.tex
python -m tccqml figuras        # results/figures/*.pdf
```

Na máquina do autor (notebook, CPU, sem GPU), `comparar` soma cerca de 7 minutos de
treino, cada `varredura` de 12 a 13, `ablacao` 3 e `diagnostico` 2; `espectro` e
`verificacoes` levam segundos. São tempos de treino, somados da coluna `segundos` de cada
CSV; o tempo de parede é um pouco maior. A `sensibilidade` não grava essa coluna: são 240
treinos, quatro vezes a grade, o que põe a sequência completa em cerca de uma hora.

Os resultados não dependem da plataforma, até onde foi conferido: em Linux x86_64, com
Python 3.13 e versões de NumPy, SciPy e scikit-learn um pouco diferentes das do lock, dois
treinos da grade (`angle`/`xor` e `zz`/`moons`, semente 43) reproduziram exatamente as
acurácias, a época escolhida e o custo final de `por_treino.csv`, e o `espectro` reproduziu
`espectro.csv`. A grade inteira só foi rodada no Windows.

## O notebook

[`notebooks/tcc_qml.ipynb`](notebooks/tcc_qml.ipynb) percorre o mesmo fluxo de forma
didática, de cima a baixo. Ele não reimplementa treino nem análise: cada experimento é uma
chamada a `python -m tccqml <subcomando>` com o mesmo Python do kernel, e cada chamada imprime
o comando equivalente. Por isso os números são os mesmos do terminal. As tabelas e figuras
exibidas são as gravadas por `tabelas` e `figuras`. O código de exibição fica em
[`tccqml/notebook.py`](src/tccqml/notebook.py), e cada célula tem uma ou duas linhas:

```python
from tccqml import notebook as nb
RODAR = False                     # True: roda os experimentos cujos arquivos faltarem
nb.iniciar(rodar=RODAR)
nb.anatomia("zz")                 # custo, circuito e estado ψ(x) de uma codificação
nb.treinar("reuploading", "moons", R=5)   # um treino, com curvas e fronteira
nb.experimento("comparar")        # um subcomando, com as tabelas e figuras do TCC
```

Há duas versões:

- [`notebooks/tcc_qml.ipynb`](notebooks/tcc_qml.ipynb) é a **interativa**, para rodar e
  mexer. É salva sem saídas.
- [`notebooks/tcc_qml_executado.ipynb`](notebooks/tcc_qml_executado.ipynb) é a **de
  leitura**: o mesmo notebook já executado, com todas as tabelas e figuras, que o GitHub
  exibe direto, sem instalar nada. Não se edita à mão: é gerada a partir da interativa por
  `python notebooks/gerar_executado.py` (cerca de 40 s, com `RODAR = False`), que deve ser
  rodado de novo sempre que a interativa ou os resultados mudarem. O script falha se alguma
  saída contiver um caminho absoluto da máquina, e o arquivo é excluído de filtros de limpeza
  como o nbstripout (`.gitattributes` e `"keep_output": true` nos metadados).

| Seção | O que tem |
|---|---|
| 0. Setup | `RODAR`, versões do Python e do PennyLane e o `listar` |
| 1. Os dados | `datasets.pdf`, as hipóteses H1–H5, bruto × normalizado, partições 180/60/60 e por que [0, π] |
| 2. Anatomia | custo com a contagem (2p + 1), circuito em blocos e `level="device"` e ψ(x) com concorrência, para a codificação escolhida |
| 3. Um treino | `treinar ... --salvar` em `results/notebook/`: custo, curvas com a época escolhida, fronteira de decisão e a comparação com a grade |
| 4. Os experimentos | uma seção por subcomando, com a pergunta, o que procurar, as tabelas de `tabelas` e as figuras de `figuras` |
| 5. Índice | cada tabela e figura, se existe e onde entra no TCC |
| 6. Terminal | a sequência completa de comandos |

**Como abrir.** Instale o extra `notebooks` (`pip install -e ".[notebooks]"`) e abra o
arquivo no **VS Code**, com o kernel da `.venv`, ou no **Jupyter**
(`jupyter lab notebooks/tcc_qml.ipynb`). O notebook acha a raiz do repositório sozinho, em
Windows ou Linux.

Cada experimento só roda se faltar algum arquivo que ele grava e se `RODAR = True`. Como os
resultados estão versionados em `results/`, o padrão só os lê, e o notebook inteiro roda em
cerca de um minuto. O treino da seção 3 não depende de `RODAR`: treinar uma combinação é uma
ação explícita e leva segundos. No Jupyter ou no VS Code, `nb.painel_treino()` e
`nb.painel_anatomia()` abrem as mesmas opções em painéis com ipywidgets.

## Onde cada número do TCC nasce

Nenhum número do Capítulo 4 é digitado à mão: todos saem de um CSV em `results/metrics/`.

| Artefato | Sai de | Produzido por |
|---|---|---|
| `tab_acuracia.tex` | `resumo.csv` | `comparar` → `tabelas` |
| `tab_custo.tex` | `resumo.csv` | `comparar` → `tabelas` |
| `tab_espectro.tex` | `espectro.csv` | `espectro` → `tabelas` |
| `tab_ablacao.tex` | `ablacao_entrelacamento.csv` | `ablacao` → `tabelas` |
| `tab_diagnostico.tex` | `diagnostico_amplitude.csv` | `diagnostico` → `tabelas` |
| `tab_mesmo_p.tex` | `varredura_R.csv` e `varredura_L_var.csv` | `varredura` (as duas) → `tabelas` |
| `tab_sensibilidade.tex` | `resumo_sensibilidade_eta.csv` | `sensibilidade` → `tabelas` |
| `tab_verificacoes.tex` | `controle_linear.csv` e `limiar_circles.csv` | `verificacoes` → `tabelas` |
| `tab_qualitativa.tex` | `QUALITATIVA`, em `tabelas.py` | `tabelas` |
| `datasets.pdf` | — | `figuras` |
| `curvas-treinamento.pdf` | `comparacao.csv` | `comparar` → `figuras` |
| `fronteiras-aprendidas.pdf` | `results/weights/` | `comparar` → `figuras` |
| `acuracia-vs-R.pdf` | `varredura_R.csv` | `varredura` → `figuras` |
| `acuracia-vs-custo.pdf` | `resumo.csv` | `comparar` → `figuras` |
| `espectro-reuploading.pdf` | medido na hora | `figuras` |
| `custo-por-codificacao.pdf` | `resumo.csv` | `comparar` → `figuras` |
| `sensibilidade-eta.pdf` | `sensibilidade_eta.csv` | `sensibilidade` → `figuras` |
| `limiar-circles.pdf` | calculado na hora (semente 42) | `figuras` |

As oito primeiras tabelas são **derivadas e não devem ser editadas à mão**: se um número
estiver estranho, o lugar de corrigir é o experimento. A `tab_qualitativa` é a exceção —
a coluna de dificuldade de implementação é julgamento, não medição. Mesmo ela não se
edita no `.tex`, que `tabelas` sobrescreve: o texto vive na lista `QUALITATIVA`, em
[`tabelas.py`](src/tccqml/tabelas.py), e é lá que se revisa.

## Análises complementares (`experiments/revisao/`)

As análises feitas depois dos resultados da versão 1.0.0 (Cap. 3, "Escolhas anteriores e
posteriores aos resultados") ficam em scripts curtos em
[`experiments/revisao/`](experiments/revisao/), que **importam** o código de `tccqml` (dados,
treino, espectro, contagem de recursos) sem alterá-lo. Tudo o que eles gravam vai para
`results/revisao/`; nenhum CSV, peso ou figura da versão 1.0.0 é sobrescrito.

Na ordem (cada script que treina aceita `--rapido`, para validar com 1 semente e 3 épocas
sem gravar nada):

```bash
python experiments/revisao/diferenca_pareada.py     # diferença pareada; só lê os CSVs da v1.0.0
python experiments/revisao/historicos_varredura.py  # varreduras com histórico e pesos: 150 treinos (~10 min)
python experiments/revisao/curvas.py                # curvas por semente, oscilações, treino e teste por k
python experiments/revisao/controle_linear.py       # controle linear com C pela validação: segundos
python experiments/revisao/controle_otimizacao.py   # eta x épocas: 480 treinos (~1 h)
python experiments/revisao/sementes_novas.py        # sementes 47-51; usa a escolha do anterior
python experiments/revisao/espectro_2d.py           # espectro 2D; usa os pesos de historicos_varredura
python experiments/revisao/recursos.py              # recursos por k: segundos
python experiments/revisao/particoes.py             # partições e verificações do pré-processamento
```

Os scripts que treinam usam `concurrent.futures` com um processo por núcleo, menos um.
Os tempos acima são de parede, na máquina do autor (8 núcleos, 7 processos).

**Versão do código.** Toda saída registra o commit que a gerou: os CSVs têm a coluna
`commit`, os PDFs levam o hash nos metadados (`Subject`), os `.npz` guardam o campo
`commit`, e cada script grava `results/revisao/versoes/<script>.json` com o hash, se `src/`
ou `experiments/` tinham alterações não commitadas, o comando e a lista de arquivos
gerados. O hash é lido com `git rev-parse HEAD`, só leitura.

**Partições.** `results/revisao/particoes/<dataset>_<semente>.npz` guarda, para as sementes
42–51, os pontos antes da normalização, os rótulos, os índices de treino, validação e teste
e os extremos do MinMaxScaler; `particoes.py` confere que eles reproduzem `load_dataset`
exatamente.

| Artefato (`results/revisao/`) | Script | Entradas |
|---|---|---|
| `figures/diferenca-pareada.pdf`, `metrics/diferenca_pareada.csv`, `metrics/resumo_diferenca_pareada.csv` | `diferenca_pareada.py` | `results/metrics/varredura_R.csv`, `varredura_L_var.csv` |
| `metrics/historico_varredura.csv`, `metrics/por_treino_varredura.csv`, `weights/varredura/*.npz`, `metrics/conferencia_varredura.csv` | `historicos_varredura.py` | treino (`tccqml.experiments.rodar_um`); conferência contra `varredura_R.csv` e `varredura_L_var.csv` |
| `figures/curvas-treinamento-sementes.pdf` | `curvas.py` | `results/metrics/comparacao.csv` |
| `figures/curvas-profundidade.pdf` | `curvas.py` | `metrics/historico_varredura.csv` |
| `tables/tab_treino_teste_profundidade.tex`, `metrics/treino_teste_profundidade.csv` | `curvas.py` | `results/metrics/varredura_R.csv`, `varredura_L_var.csv` |
| `metrics/oscilacoes.csv` | `curvas.py` | `comparacao.csv`, `por_treino.csv`, `metrics/historico_varredura.csv` |
| `tables/tab_controle_linear_C.tex`, `metrics/controle_linear_C.csv`, `metrics/resumo_controle_linear_C.csv` | `controle_linear.py` | dados gerados por `load_dataset` (sementes 42–46) |
| `tables/tab_controle_otimizacao.tex`, `metrics/controle_otimizacao.csv`, `metrics/resumo_controle_otimizacao.csv`, `metrics/validacao_controle_otimizacao.csv`, `metrics/historico_controle_otimizacao.csv`, `metrics/conferencia_otimizacao.csv` | `controle_otimizacao.py` | treino; conferência contra `varredura_*.csv` e `sensibilidade_eta.csv` |
| `tables/tab_sementes_novas.tex`, `metrics/sementes_novas.csv`, `metrics/resumo_sementes_novas.csv`, `metrics/*diferenca_pareada_sementes_novas.csv` | `sementes_novas.py` | `metrics/resumo_controle_otimizacao.csv`; treino com sementes 47–51 |
| `figures/espectro-2d.pdf`, `tables/tab_espectro_2d.tex`, `metrics/espectro_2d.csv`, `metrics/resumo_espectro_2d.csv`, `metrics/coeficientes_espectro_2d.csv` | `espectro_2d.py` | `weights/varredura/*.npz` |
| `tables/tab_recursos_profundidade.tex`, `metrics/recursos_profundidade.csv` | `recursos.py` | `tccqml.circuit_stats.stats` |
| `particoes/*.npz`, `metrics/verificacao_particoes.csv` | `particoes.py` | dados gerados por `load_dataset` (sementes 42–51) |

As figuras e tabelas citadas no texto são copiadas, sem edição, de `results/revisao/figures/`
e `results/revisao/tables/` para `figuras/revisao/` e `tabelas/revisao/` do projeto LaTeX.

## O protocolo congelado

Vive inteiro em [`src/tccqml/config.py`](src/tccqml/config.py), e é a fonte fiel para
escrever o Capítulo 3:

```
ansatz        StronglyEntanglingLayers, L_var = 2   (n = 2 -> p = 12, 3 deles inertes)
init          N(0, 0.1²), semente fixa
otimizador    Adam, eta = 0.1, 30 épocas, lotes de 20
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

Com dois qubits e leitura de Z_0, o par de CNOTs no fim de cada camada leva Z_0 a Z_1, e a
`Rot` da última camada no qubit 0 não afeta a saída: 3 dos 12 parâmetros da configuração de
referência têm derivada identicamente nula. Eles continuam contados em `p`, porque fazem
parte do circuito e seriam avaliados pelo *parameter-shift*. Pelo mesmo motivo, no
re-uploading a frequência máxima em x_1 é R − 1, e com uma única camada (L_var = 1 ou R = 1)
a saída não depende de x_1. Os testes `test_par_de_cnots_leva_z0_a_z1`,
`test_rot_final_do_qubit_0_e_inerte` e `test_uma_camada_ignora_o_primeiro_atributo` conferem
os três fatos.

Exceções deliberadas, todas reportadas como tais:

1. o ansatz `local` (sem CNOTs) existe só para a ablação da hipótese H3;
2. `varredura --parametro L_var` muda as camadas do ansatz no angle, como controle da
   varredura de R (mesmo `p`, espectro fixo); grava num CSV próprio;
3. o `amplitude` usa 1 qubit em vez de 2, porque a codificação é assim — e por isso `p`
   cai de 12 para 6;
4. o `reuploading` não aplica o ansatz depois dos dados: intercala R camadas com os
   blocos de dados, e por isso tem `p = 6R` (18 com R = 3).
5. `sensibilidade` repete a grade variando só o eta sobre `eta_candidatos` (0.01, 0.03, 0.1,
   0.3, fixados antes de rodar; 4 valores x 5 sementes para todas as codificações), escolhe o
   eta pela maior acurácia média de validação, sem olhar o teste, e o reporta ao lado do
   eta = 0.1 do protocolo: separa falha de otimização de limitação de representação.

As tabelas que comparam codificações informam `n_qubits` e `p` (na nota da `tab_acuracia`
e nas colunas da `tab_custo`), para a comparação não ser lida como se fosse de parâmetros
iguais.

### Verificações sobre os dados (`verificacoes`)

Duas checagens clássicas, sem circuito, com as mesmas partições, normalização e sementes da
grade; o teste só entra na avaliação final. Gravam em CSVs próprios.

- **Controle linear.** Regressão logística (C = 1, `max_iter` = 1000, fixados antes de rodar)
  sobre as nove funções {1, cos x_j, sin x_j} e seus produtos, nas coordenadas em [0, pi].
  Diz se esse espaço contém um separador — não que o circuito consiga realizá-lo ou aprendê-lo.
- **Limiar de g no circles.** A regra g(x)/2 > t, com g(x) = sin x1 + sin x2,
  limiar e sentido escolhidos no treino e avaliados em validação e teste, sem ruído e com o
  ruído do protocolo. Registra também
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
  notebook.py       apoio ao notebook: só exibe, não treina nem analisa
tests/              208 testes
notebooks/          tcc_qml.ipynb: o fluxo inteiro da CLI, de forma didática;
                    tcc_qml_executado.ipynb: a cópia executada, para leitura
                    (gerada por gerar_executado.py)
results/metrics/    CSVs de cada experimento
results/tables/     .tex (e .csv) para o Overleaf
results/figures/    .pdf para o Overleaf
results/weights/    pesos finais da grade (.npz), de onde saem as fronteiras
                    (as quatro pastas são versionadas; results/notebook/, não)
experiments/revisao/  scripts das análises complementares (importam tccqml)
results/revisao/    o que eles gravam: metrics, tables, figures, weights,
                    particoes e versoes (versionado; logs/, não)
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
  e o re-uploading com R = 3 tem 18.
- **Cinco sementes.** Diferença de acurácia menor que o desvio reportado não sustenta
  afirmação.
- **Basis encoding e embeddings treináveis ficam como teoria, sem experimento.** É escolha
  de escopo declarada, não omissão, com a justificativa do Cap. 2 e do Cap. 3: o *basis*
  exige discretizar os atributos, uma decisão a mais que muda o problema, e multiplica os
  qubits (16 para dois atributos com 8 bits cada); nos *embeddings* treináveis, parâmetros
  treináveis entram na própria codificação, a representação dos dados deixa de ser fixa, e
  uma diferença de desempenho não poderia ser atribuída só à forma de codificar.

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
