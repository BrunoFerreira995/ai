# Contribuindo

Obrigado por contribuir. Antes de abrir uma alteração:

1. Crie uma branch pequena e descritiva.
2. Ative o ambiente com `source .venv/bin/activate`.
3. Execute os testes:

   ```bash
   .venv/bin/python -m unittest discover
   ```

4. Atualize a documentação quando mudar comandos, formatos ou comportamento.
5. Abra um pull request descrevendo objetivo, testes executados e impactos nos
   artefatos de modelo.

Não inclua datasets privados, chaves, tokens, `.venv` ou artefatos grandes sem
necessidade. Para mudanças no modelo, registre dataset, seed, hiperparâmetros,
métricas e formato exportado.
