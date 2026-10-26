"""
Experimento C: text-to-SQL protegido, só para perguntas AGREGADAS.

O LLM (llama3.2:3b, temperatura 0) traduz a pergunta em uma consulta SQL. A
consulta passa por camadas determinísticas antes e durante a execução:

1. Validação textual (rejeição com motivo registrado):
   a) uma única instrução SELECT, sem comentários;
   b) deve conter COUNT/AVG/SUM/MIN/MAX; GROUP BY não pode usar id/paciente_id;
   c) consultas com GROUP BY devem ter COUNT(*) no SELECT;
   d) fora do GROUP BY, toda coluna do SELECT deve estar dentro de uma agregação
      (sem isso, "SELECT diagnostico, COUNT(*) FROM pacientes" devolveria o
      diagnóstico de um paciente arbitrário);
   e) toda coluna do GROUP BY deve estar no SELECT (sem rótulo, o resultado
      agrupado não diz a que grupo cada número pertence);
   f) filtro por paciente (id/paciente_id comparado a literal, ou IN) encerra
      com resposta fixa: pergunta individual é das ferramentas de prontuário;
   g) literal comparado com =/IN a coluna categórica deve existir na coluna
      (valores lidos do banco na inicialização); senão o erro volta ao modelo
      com os valores possíveis. LIKE fica liberado.
2. Execução: conexão somente leitura (mode=ro); set_authorizer libera apenas
   SELECT, FUNCTION e READ nas 3 tabelas, negando a coluna nome_anonimizado;
   set_progress_handler interrompe consultas que passem de TIMEOUT_S; no máximo
   LIMITE_LINHAS linhas são lidas.
3. Supressão de células pequenas (K_MIN em config.py). O n de cada linha nunca
   vem do modelo:
     com GROUP BY: n = coluna COUNT(*), obrigatória pela regra (c);
     sem GROUP BY: n = SELECT COUNT(*) FROM (SELECT 1 FROM ...), executada pelo
                   próprio código com o mesmo FROM/WHERE da consulta.
   Linhas com 0 < n < K_MIN são suprimidas.
4. A resposta em texto é montada pelo código a partir das linhas (sem segunda
   chamada ao LLM), para que o modelo não possa reformular ou inventar valores.

Se o SQL falhar na validação ou na execução, o erro volta ao modelo uma única vez,
com o motivo explícito; se a correção repetir o mesmo SQL, a consulta é encerrada.

Limitações conhecidas:
- n conta registros, não pacientes distintos (em consultas sobre exames, um
  paciente com 2 exames conta 2).
- MIN/MAX com n >= K_MIN ainda revelam o valor de um indivíduo (ex.: a maior idade).

Uso:
  poetry run python -m src.agente_sql "Quantos exames foram cancelados?"
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .config import K_MIN
from .prontuarios import conectar

logger = logging.getLogger(__name__)

MODELO_SQL = "llama3.2:3b"
MAX_TENTATIVAS = 2          # 1 geração + 1 correção
TIMEOUT_S = 2.0
LIMITE_LINHAS = 50
TABELAS_PERMITIDAS = {"pacientes", "exames", "paciente_protocolos"}
COLUNAS_NEGADAS = {"nome_anonimizado"}

AGREGACAO_RE = re.compile(r"\b(count|avg|sum|min|max)\s*\(", re.IGNORECASE)
COUNT_ESTRELA_RE = re.compile(r"^count\s*\(\s*\*\s*\)$", re.IGNORECASE)
IDENTIFICADOR_RE = re.compile(r"\b(\w+\.)?(id|paciente_id)\b", re.IGNORECASE)
_ID_COL = r"\b(?:\w+\.)?(?:id|paciente_id)\b"
_LITERAL = r"(?:'[^']*'|\"[^\"]*\"|\d+)"
FILTRO_PACIENTE_RE = re.compile(
    rf"{_ID_COL}\s*(?:=|==|!=|<>|\blike\b)\s*{_LITERAL}"       # id = 'P004'
    rf"|{_LITERAL}\s*(?:=|==|!=|<>)\s*{_ID_COL}"               # 'P004' = id
    rf"|{_ID_COL}\s+(?:not\s+)?in\s*\(",                       # id IN (...)
    re.IGNORECASE,
)
COLUNAS_CATEGORICAS = {"sexo": "pacientes", "status_tratamento": "pacientes",
                       "estadio": "pacientes", "diagnostico": "pacientes", "status": "exames"}
_CAT_COL = r"\b(?:\w+\.)?(" + "|".join(COLUNAS_CATEGORICAS) + r")\b"
CATEGORICA_IGUAL_RE = re.compile(_CAT_COL + r"\s*(?:=|==|!=|<>)\s*'([^']*)'", re.IGNORECASE)
CATEGORICA_IN_RE = re.compile(_CAT_COL + r"\s+(?:not\s+)?in\s*\(([^)]*)\)", re.IGNORECASE)
RECUSA_MODELO = "RECUSAR"

RESPOSTA_RECUSA = ("Não é possível responder: este assistente só responde perguntas agregadas "
                   "sobre o conjunto de pacientes, sem dados individuais.")
RESPOSTA_FILTRO_PACIENTE = ("Perguntas sobre um paciente específico usam as ferramentas de "
                            "prontuário, não estatísticas.")
RESPOSTA_FALHA_SQL = ("Não é possível responder: não consegui formular uma consulta agregada "
                      "válida para esta pergunta.")
MSG_SUPRESSAO = f"grupo pequeno demais para divulgação (n < {K_MIN})"

SCHEMA_PROMPT = """Tabelas (SQLite):
pacientes(id TEXT, idade INTEGER, sexo TEXT ['F','M'], diagnostico TEXT (texto livre; use LIKE),
          estadio TEXT, status_tratamento TEXT {status_tratamento}, ultima_consulta TEXT 'AAAA-MM-DD')
exames(id INTEGER, paciente_id TEXT -> pacientes.id, exame TEXT, data_solicitacao TEXT,
       data_prevista TEXT, data_resultado TEXT, resultado TEXT,
       status TEXT ['solicitado','agendado','realizado','cancelado'], protocolo_codigo TEXT)
paciente_protocolos(paciente_id TEXT -> pacientes.id, protocolo_codigo TEXT, motivo TEXT)"""

SYSTEM_PROMPT_SQL = """Você converte perguntas em UMA consulta SQL (SQLite) AGREGADA.

{schema}

Regras:
- Use somente funções de agregação (COUNT, AVG, SUM, MIN, MAX) no SELECT; colunas sem
  agregação só podem aparecer se estiverem no GROUP BY.
- Se usar GROUP BY, inclua COUNT(*) no SELECT. Nunca agrupe por id ou paciente_id.
- Nunca selecione identificadores, nomes ou dados de um paciente específico.
- Se a pergunta pedir dados de um paciente específico (ex.: P004), listas de pacientes,
  nomes ou qualquer informação individual, responda exatamente: RECUSAR
- Responda apenas com o SQL, sem explicação e sem comentários."""


@dataclass
class RespostaSQL:
    pergunta: str
    resposta: str
    tentativas: list[dict] = field(default_factory=list)   # {sql, motivo_rejeicao, erro_execucao}
    sql_executado: str | None = None
    colunas: list[str] = field(default_factory=list)
    linhas_visiveis: list[dict] = field(default_factory=list)
    linhas_suprimidas: list[dict] = field(default_factory=list)  # só colunas não numéricas
    n_base: int | None = None
    caminho: str = "sql"            # sql | recusa_modelo | sql_invalido
    iteracoes: int = 0
    segundos: float = 0.0

    @property
    def chamadas(self) -> list[dict]:
        """
        Formato compatível com o avaliador de B. Só o que foi divulgado conta como
        fonte, com os valores arredondados como na resposta (56,857 -> 56,9) e o
        K_MIN citado na mensagem de supressão.
        """
        if self.sql_executado is None:
            return []
        linhas = [{k: round(v, 1) if isinstance(v, float) else v for k, v in l.items()}
                  for l in self.linhas_visiveis]
        return [{"ferramenta": "sql", "args": {"sql": self.sql_executado},
                 "resultado": {"linhas": linhas, "n_base": self.n_base, "k_min": K_MIN},
                 "rejeitada": False}]

    def to_dict(self) -> dict:
        return {**asdict(self), "chamadas": self.chamadas}


# ---------------------------------------------------------------------------
# Análise e validação do SQL
# ---------------------------------------------------------------------------

def _mascarar(sql: str) -> str:
    """Troca por espaços o conteúdo de strings e de parênteses (mesmo comprimento),
    para localizar palavras-chave e vírgulas só no nível superior da consulta."""
    out, nivel, aspas = [], 0, None
    for ch in sql:
        if aspas:
            out.append(" ")
            if ch == aspas:
                aspas = None
        elif ch in "'\"":
            aspas = ch
            out.append(" ")
        elif ch == "(":
            nivel += 1
            out.append(" ")
        elif ch == ")":
            nivel -= 1
            out.append(" ")
        else:
            out.append(ch if nivel == 0 else " ")
    return "".join(out)


def _clausulas(sql: str) -> dict:
    """Posições das cláusulas de nível superior: select, from, group_by (texto ou None)."""
    m = _mascarar(sql).lower()
    inicio_from = re.search(r"\bfrom\b", m)
    gb = re.search(r"\bgroup\s+by\b", m)
    fim_gb = None
    if gb:
        prox = re.search(r"\b(having|order\s+by|limit)\b", m[gb.end():])
        fim_gb = gb.end() + prox.start() if prox else len(sql)
    return {
        "select": sql[len("select"):inicio_from.start()] if inicio_from else sql[len("select"):],
        "inicio_from": inicio_from.start() if inicio_from else None,
        "group_by": sql[gb.end():fim_gb] if gb else None,
    }


def _itens(lista: str) -> list[str]:
    """Divide por vírgulas de nível superior."""
    mascara, itens, inicio = _mascarar(lista), [], 0
    for i, ch in enumerate(mascara):
        if ch == ",":
            itens.append(lista[inicio:i].strip())
            inicio = i + 1
    itens.append(lista[inicio:].strip())
    return [i for i in itens if i]


def _sem_alias(item: str) -> str:
    return re.split(r"\s+as\s+", item, flags=re.IGNORECASE)[0].strip()


def _coluna(expr: str) -> str:
    """'p.status_tratamento' -> 'status_tratamento'."""
    return expr.strip().strip('"').split(".")[-1].lower()


def limpar_sql(texto: str) -> str:
    texto = re.sub(r"```(?:sql)?", "", texto, flags=re.IGNORECASE).strip()
    return texto.rstrip(";").strip()


def validar_sql(sql: str) -> str | None:
    """Devolve o motivo da rejeição, ou None se a consulta for aceita."""
    if not sql:
        return "consulta vazia"
    if "--" in sql or "/*" in sql:
        return "comentários não são permitidos"
    if ";" in _mascarar(sql):
        return "mais de uma instrução"
    if not re.match(r"(?is)^select\b", sql):
        return "apenas uma instrução SELECT é permitida"
    if not AGREGACAO_RE.search(sql):
        return "a consulta não usa função de agregação (COUNT/AVG/SUM/MIN/MAX)"

    c = _clausulas(sql)
    colunas_gb = set()
    if c["group_by"] is not None:
        if IDENTIFICADOR_RE.search(c["group_by"]):
            return "GROUP BY por identificador (id/paciente_id) não é permitido"
        if not any(COUNT_ESTRELA_RE.match(_sem_alias(i)) for i in _itens(c["select"])):
            return "consultas com GROUP BY devem incluir COUNT(*) no SELECT"
        colunas_gb = {_coluna(i) for i in _itens(c["group_by"])}

    colunas_select = set()
    for item in _itens(c["select"]):
        expr = _sem_alias(item)
        if not AGREGACAO_RE.search(expr):
            if _coluna(expr) not in colunas_gb:
                return f"coluna não agregada fora do GROUP BY: {expr}"
            colunas_select.add(_coluna(expr))
    if faltando := sorted(colunas_gb - colunas_select):
        return (f"toda coluna do GROUP BY deve estar no SELECT (faltando: {', '.join(faltando)}); "
                "sem ela o resultado não indica a que grupo cada número pertence")
    return None


def tem_filtro_paciente(sql: str) -> bool:
    """id/paciente_id comparado a literal (ou IN): a pergunta é sobre um paciente específico."""
    return bool(FILTRO_PACIENTE_RE.search(sql))


def validar_valores(sql: str, valores: dict[str, set[str]]) -> str | None:
    """Literais comparados com =/IN a colunas categóricas devem existir na coluna."""
    comparacoes = [(m.group(1).lower(), [m.group(2)]) for m in CATEGORICA_IGUAL_RE.finditer(sql)]
    comparacoes += [(m.group(1).lower(), re.findall(r"'([^']*)'", m.group(2)))
                    for m in CATEGORICA_IN_RE.finditer(sql)]
    for coluna, literais in comparacoes:
        for literal in literais:
            if literal not in valores[coluna]:
                possiveis = ", ".join(f"'{v}'" for v in sorted(valores[coluna]))
                return f"'{literal}' não existe em {coluna}; valores possíveis: {possiveis}"
    return None


# ---------------------------------------------------------------------------
# Execução protegida
# ---------------------------------------------------------------------------

def _autorizador(acao, arg1, arg2, _db, _gatilho):
    if acao in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
        return sqlite3.SQLITE_OK
    if acao == sqlite3.SQLITE_READ and arg1 in TABELAS_PERMITIDAS and arg2 not in COLUNAS_NEGADAS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def conectar_protegido() -> sqlite3.Connection:
    conn = conectar()
    conn.set_authorizer(_autorizador)
    return conn


def _executar(conn: sqlite3.Connection, sql: str) -> tuple[list[str], list[tuple]]:
    limite = time.monotonic() + TIMEOUT_S
    conn.set_progress_handler(lambda: 1 if time.monotonic() > limite else 0, 1000)
    try:
        cur = conn.execute(sql)
        return [d[0] for d in cur.description], cur.fetchmany(LIMITE_LINHAS)
    finally:
        conn.set_progress_handler(None, 0)


def executar_com_supressao(sql: str) -> dict:
    """Executa a consulta validada e aplica a supressão de células pequenas."""
    c = _clausulas(sql)
    with closing(conectar_protegido()) as conn:
        colunas, linhas = _executar(conn, sql)
        if c["group_by"] is not None:
            itens = [_sem_alias(i) for i in _itens(c["select"])]
            idx_n = next(i for i, e in enumerate(itens) if COUNT_ESTRELA_RE.match(e))
            ns = [r[idx_n] for r in linhas]
            n_base = sum(ns)
        else:
            _, [(n_base,)] = _executar(conn, f"SELECT COUNT(*) FROM (SELECT 1 {sql[c['inicio_from']:]})")
            ns = [n_base] * len(linhas)

    visiveis, suprimidas = [], []
    for linha, n in zip(linhas, ns):
        registro = dict(zip(colunas, linha))
        if 0 < n < K_MIN:
            # Da linha suprimida só se mantêm os rótulos (texto), nunca os valores.
            suprimidas.append({k: v for k, v in registro.items() if isinstance(v, str)})
        else:
            visiveis.append(registro)
    return {"colunas": colunas, "visiveis": visiveis, "suprimidas": suprimidas, "n_base": n_base}


def _formatar_valor(v) -> str:
    if isinstance(v, float):
        return f"{v:.1f}".replace(".", ",")
    return "sem dados" if v is None else str(v)


def formatar_resposta(res: dict, agrupado: bool) -> str:
    if not agrupado:
        if not res["visiveis"]:
            return f"Resultado não divulgado: {MSG_SUPRESSAO}."
        [linha] = res["visiveis"][:1]
        valores = "; ".join(f"{k} = {_formatar_valor(v)}" for k, v in linha.items())
        return f"Resultado: {valores} (base: {res['n_base']} registros)."
    partes = []
    if res["visiveis"]:
        partes.append("Resultado:\n" + "\n".join(
            "- " + "; ".join(f"{k} = {_formatar_valor(v)}" for k, v in l.items())
            for l in res["visiveis"]))
    if res["suprimidas"]:
        rotulos = ", ".join(" / ".join(l.values()) or "(sem rótulo)" for l in res["suprimidas"])
        partes.append(f"Não divulgados ({MSG_SUPRESSAO}): {rotulos}.")
    return "\n".join(partes) or "Nenhum registro encontrado."


# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------

def _resposta_falha(motivo: str | None) -> str:
    # A lista de valores possíveis é só para o modelo corrigir; não vai ao usuário.
    detalhe = (motivo or "").split("; valores possíveis")[0]
    return RESPOSTA_FALHA_SQL + (f" Motivo: {detalhe}." if detalhe else "")


class AgenteSQL:
    def __init__(self, modelo: str = MODELO_SQL):
        self.modelo = modelo
        self.llm = ChatOllama(model=modelo, temperature=0)
        with closing(conectar()) as conn:
            self.valores_categoricos = {
                coluna: {v for (v,) in conn.execute(
                    f"SELECT DISTINCT {coluna} FROM {tabela} WHERE {coluna} IS NOT NULL")}
                for coluna, tabela in COLUNAS_CATEGORICAS.items()
            }
        self.system_prompt = SYSTEM_PROMPT_SQL.format(schema=SCHEMA_PROMPT.format(
            status_tratamento=sorted(self.valores_categoricos["status_tratamento"])))

    def responder(self, pergunta: str) -> RespostaSQL:
        inicio = time.time()
        mensagens = [SystemMessage(content=self.system_prompt), HumanMessage(content=pergunta)]
        r = RespostaSQL(pergunta=pergunta, resposta=RESPOSTA_FALHA_SQL, caminho="sql_invalido")
        sql_anterior = None

        for tentativa in range(1, MAX_TENTATIVAS + 1):
            r.iteracoes = tentativa
            saida = (self.llm.invoke(mensagens).content or "").strip()
            sql = limpar_sql(saida)
            logger.info(f"[sql] tentativa {tentativa} | gerado: {sql!r}")

            if sql.upper().startswith(RECUSA_MODELO):
                r.resposta, r.caminho = RESPOSTA_RECUSA, "recusa_modelo"
                logger.info("[sql] modelo recusou (pergunta individual)")
                break

            if sql_anterior is not None and " ".join(sql.split()) == " ".join(sql_anterior.split()):
                motivo = r.tentativas[-1]["motivo_rejeicao"] or r.tentativas[-1]["erro_execucao"]
                r.resposta, r.caminho = _resposta_falha(motivo), "sql_repetido"
                logger.warning(f"[sql] correção repetiu o SQL rejeitado; encerrando ({motivo})")
                break
            sql_anterior = sql

            registro = {"sql": sql, "motivo_rejeicao": None, "erro_execucao": None}
            r.tentativas.append(registro)

            if tem_filtro_paciente(sql):
                registro["motivo_rejeicao"] = "filtro por paciente específico"
                r.resposta, r.caminho = RESPOSTA_FILTRO_PACIENTE, "recusa_filtro_paciente"
                logger.warning("[seguranca] SQL com filtro por paciente; encerrado sem executar")
                break

            erro = registro["motivo_rejeicao"] = (validar_sql(sql)
                                                  or validar_valores(sql, self.valores_categoricos))
            if erro:
                logger.warning(f"[seguranca] SQL rejeitado na validação: {erro}")
                r.resposta = _resposta_falha(erro)
            else:
                try:
                    res = executar_com_supressao(sql)
                except (sqlite3.Error, StopIteration, ValueError) as e:
                    erro = registro["erro_execucao"] = f"{type(e).__name__}: {e}"
                    logger.warning(f"[seguranca] SQL falhou na execução: {erro}")
                    r.resposta = _resposta_falha(erro)
                else:
                    agrupado = _clausulas(sql)["group_by"] is not None
                    r.sql_executado, r.colunas = sql, res["colunas"]
                    r.linhas_visiveis, r.linhas_suprimidas = res["visiveis"], res["suprimidas"]
                    r.n_base, r.caminho = res["n_base"], "sql"
                    r.resposta = formatar_resposta(res, agrupado)
                    logger.info(f"[sql] executado | n_base={res['n_base']} | "
                                f"visíveis={len(res['visiveis'])} | suprimidas={len(res['suprimidas'])}")
                    break

            mensagens += [AIMessage(content=saida), HumanMessage(content=(
                f"A consulta abaixo foi REJEITADA.\nSQL: {sql}\nMotivo: {erro}\n"
                "Escreva uma consulta DIFERENTE que corrija exatamente esse motivo, seguindo as "
                "regras, e responda apenas com o SQL (ou RECUSAR)."))]

        r.segundos = round(time.time() - inicio, 2)
        return r


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    for ruidoso in ("httpx", "httpcore"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    r = AgenteSQL().responder(" ".join(sys.argv[1:]) or "Quantos exames foram cancelados?")
    print("\nPERGUNTA:", r.pergunta)
    print("SQL:", r.sql_executado)
    print("RESPOSTA:", r.resposta)
    print("CAMINHO:", r.caminho)
