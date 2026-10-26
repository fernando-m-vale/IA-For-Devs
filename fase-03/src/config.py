"""
Configurações compartilhadas da Fase 3.

DATA_REFERENCIA é o "hoje" fixo dos dados sintéticos: as regras de pendência
usam esta data, e não a data do sistema, para que os resultados sejam
reproduzíveis.

Regras de pendência de exames (tabela `exames` de prontuarios.db):
- pendente: status em STATUS_PENDENTES ('solicitado', 'agendado').
- atrasado: pendente E data_prevista < DATA_REFERENCIA (estritamente menor).
  Um exame previsto para o próprio dia da referência é pendente, não atrasado.
  Exame pendente sem data_prevista não é considerado atrasado.
"""

DATA_REFERENCIA = "2026-10-20"
STATUS_PENDENTES = ("solicitado", "agendado")

# Supressão de células pequenas: resultados agregados que representem de 1 a
# K_MIN - 1 registros não são divulgados (risco de reidentificação).
K_MIN = 3
