# Changelog

As versões seguem as tags do repositório. A divisão entre elas é a do Cap. 3 ("Escolhas
anteriores e posteriores aos resultados"): a 1.0.0 contém o que foi fixado antes dos
resultados, e a 1.1.0 acrescenta o que foi definido depois deles.

## 1.1.0

Análises definidas depois dos resultados, pedidas na revisão do texto. Nada do código de
`src/tccqml/` que gera resultados mudou, e nenhum arquivo de `results/metrics`, `tables`,
`figures` ou `weights` foi alterado: tudo o que é novo grava em `results/revisao/`.

### Adicionado

- Scripts em `experiments/revisao/`, que importam `tccqml` sem alterá-lo:
  - `diferenca_pareada.py`: diferença pareada por semente entre o re-uploading com R = k e o
    angle com L_var = k (Figura 5);
  - `historicos_varredura.py`: as 150 execuções das varreduras de R e de L_var de novo, agora
    com histórico por época e pesos; conferem com a 1.0.0 com diferença máxima inferior a
    10⁻¹⁵;
  - `curvas.py`: curvas por semente (Figuras 3 e 6), oscilações nas dez últimas épocas,
    limiar relativo de 90% e acurácias de treino e de teste por profundidade (Tabela 13);
  - `controle_linear.py`: controle linear com C escolhido pela validação (Tabela 17);
  - `controle_otimizacao.py`: grade de η × épocas nas configurações da varredura de R
    (Tabela 19); as 100 comparações com experimentos anteriores conferem com diferença
    máxima inferior a 10⁻¹⁵;
  - `sementes_novas.py`: as mesmas configurações nas sementes 47 a 51 (Tabela 20);
  - `espectro_2d.py`: espectro permitido, alcançável e usado para k = 1 a 5 e nos modelos
    treinados (Tabela 10 e Figura 2);
  - `recursos.py`: recursos do circuito compilado por profundidade (Tabela 8);
  - `particoes.py`: partições de cada conjunto e semente e verificações do pré-processamento.
- Saídas em `results/revisao/`, com o commit que as gerou em cada CSV, PDF e `.npz` e um
  manifesto por script em `results/revisao/versoes/`.
- `test_parametros_inertes`: confere que 5 dos 12 parâmetros da configuração de referência
  (4 dos 6 com uma camada) têm derivada identicamente nula.
- `CITATION.cff`.

### Corrigido

- O histórico do controle de otimização é ordenado antes de ser gravado, para que o CSV não
  dependa da ordem em que os processos terminam.
- A docstring de `ansatz.py` e o README diziam que 3 dos 12 parâmetros são inertes; são 5,
  como no texto.

### Documentação

- O README liga cada tabela e figura do texto ao arquivo, aos dados e ao comando que a
  produzem, e indica quais saídas da CLI ficaram fora do texto final.
- O índice do notebook usa as seções e os rótulos LaTeX do texto final.
- Limitações do estudo alinhadas com o Cap. 5.
- `texto/`, o projeto LaTeX local, fica fora do git.

## 1.0.0

Protocolo e experimentos fixados antes dos resultados.

- Protocolo congelado em `src/tccqml/config.py`: `StronglyEntanglingLayers` com L_var = 2,
  Adam com η = 0,1, 30 épocas, lotes de 20, custo quadrático, leitura de Z_0, N = 300,
  divisão 180/60/60 com parada pela validação e sementes 42 a 46.
- Quatro codificações (angle, amplitude, data re-uploading e feature map ZZ) e o ansatz sem
  CNOTs da ablação.
- CLI `tccqml` com os subcomandos `listar`, `treinar`, `comparar`, `sensibilidade`, `varredura`,
  `espectro`, `ablacao`, `diagnostico`, `verificacoes`, `tabelas` e `figuras`.
- Resultados versionados em `results/metrics`, `tables`, `figures` e `weights`.
- Notebook interativo e cópia executada para leitura.
- Lock de dependências (`requirements-lock.txt`) e licença MIT.
