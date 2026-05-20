# Changelog — edt.py

Todas as mudanças relevantes do `edt.py` estão documentadas aqui.
Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

---

## [2.0.0] — 2025-05-19

### Resumo
Grande refatoração para suporte a interface gráfica (GUI), correção de bugs
críticos e melhoria na organização do código. O algoritmo de digitalização
do ECG permanece o mesmo — apenas a estrutura do módulo foi modernizada.

---

### Adicionado
- **Função auxiliar `_show(fig, plot_callback)`**: redireciona cada figura
  matplotlib para o callback da GUI ou para `plt.show()` no modo standalone,
  e sempre fecha a figura para liberar memória.
- **Chave `plot_callback`** no `config_dict`: quando preenchida com um callable,
  todos os gráficos de debug são enviados para a GUI em vez de bloquear o terminal.
- **Chave `show_images`** no `config_dict`: flag booleana que suprime todos os
  gráficos de debug quando `False` (padrão), independentemente do nível de `verbose`.
- Import de `pytesseract` e de `remove_text` a partir de `edt_utils`.
- **Docstring completa** em `ecg_to_csv()` documentando cada parâmetro do `config_dict`.
- `plot_callback` repassado para as subfunções `display_segments()`,
  `process_line()` e `print_line_dict()`.
- `pulse_per_sec`, `pulse_per_mv` e `num_sampling_points` agora desempacotados
  explicitamente do `config_dict` dentro da função (antes dependiam de variáveis globais).
- `sliced_labeled_line = labeled_line.copy()` adicionado no bloco `else` quando
  a linha não possui pulso de calibração, evitando `NameError` em tempo de execução.
- Retorno de `ecg_to_csv()` ampliado com artefatos intermediários úteis para
  debug e exibição na GUI:
  ```python
  {"df": ecg_df, "foreground": foreground, "binary": th1, "image_gray": image_gray}
  ```

---

### Alterado
- A chamada de `ecg_to_csv()` agora está dentro de `if __name__ == "__main__"`,
  de modo que importar o módulo a partir de uma GUI não dispara o processamento.
- Os três blocos de estratégia de pré-processamento foram convertidos de
  `if` / `if` / `if` independentes para `if` / `elif` / `else`, evitando
  execução dupla.
- `wt` e `ht` (dimensões do pulso de calibração) agora são calculados
  **uma única vez antes do loop**, em vez de a cada iteração.
- `detect_ref_pulse()` chamado com threshold padrão (`0.6`) em vez do
  valor fixo `threshold=0.4` da versão anterior.
- Todas as chamadas de debug com `plt.imshow()` / `plt.show()` agora estão
  protegidas por `show_images and verbose > N` e usam `_show(fig, plot_callback)`.
- Dois blocos redundantes de plot do foreground (um em `verbose > 1` e outro
  em `verbose > 0`) foram removidos.
- `config_dict` reescrito como um único dicionário literal — chaves duplicadas
  (`pulse_width_mm`, `pulse_height_mm`, `pulse_per_mv`, `pulse_per_sec`)
  que eram atribuídas duas vezes agora aparecem uma única vez.
- `num_sampling_points` corrigido de `time_lead / (1/sample_frequency)` (float)
  para `int(time_lead * sample_frequency)` (inteiro).
- Nomes de arquivo padrão atualizados: `'bucket/img20250221_05271756'` → `'ecg2'`,
  `'bucket/template.png'` → `'pul.png'`.
- `verbose` padrão alterado de `3` (muito verboso) para `0` (silencioso).
- Duas linhas separadas de `from edt_utils import ...` unificadas em um único
  import agrupado.
- Variável não utilizada `line_copy_width, line_copy_height` removida.

---

### Corrigido
- **Crítico — processo encerrado em caminho de imagem inválido**: `sys.exit(0)`
  substituído por `raise ValueError(...)`, permitindo que a GUI capture e trate
  o erro sem encerrar o processo.
- **Crítico — `NameError` em linhas sem pulso de calibração**: `sliced_labeled_line`
  não era inicializado quando a linha não possuía pulso, causando falha na
  chamada subsequente de `process_line()`.
- **Typo silencioso na chave do `config_dict`**: `config_dict['pulse_per_sec ']`
  (espaço extra ao final) corrigido para `config_dict['pulse_per_sec']`. A chave
  errada era ignorada silenciosamente e a função usava uma variável global como fallback.
- **Chaves duplicadas no `config_dict`**: `pulse_width_mm`, `pulse_height_mm`,
  `pulse_per_mv` e `pulse_per_sec` eram atribuídos duas vezes; a última escrita
  vencia, mascarando o valor anterior.
- **Verificação de sanidade do template testava a variável errada**: `if image is None`
  era usado para checar se o template carregou — corrigido para `if template is None`.

---

### Removido
- Bloco comentado de declarações de variáveis antigas no topo de `ecg_to_csv()`
  (sobras de refatoração anterior).
- Comentários `# TODO` já resolvidos.
- Variável `blank_image` alocada mas nunca utilizada.
- Lista `slices_y` calculada mas nunca utilizada.

---

## [1.0.0] — 2025-04-27

Versão inicial em funcionamento (`edt_original_27_04_26.py`).
