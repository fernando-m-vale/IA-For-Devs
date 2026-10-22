"""
Acesso somente leitura aos prontuários (data/synthetic/prontuarios.db).

Funções usadas como ferramentas pelo agente: recebem e devolvem apenas tipos
serializáveis em JSON. A conexão é aberta em modo somente leitura (URI
mode=ro), então nenhuma ferramenta consegue alterar o banco.

Minimização de dados: nenhuma função devolve nome_anonimizado. O agente
identifica pacientes só pelo código (P000); um nome que não chega ao modelo
não pode ser revelado por ele.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path

from .config import DATA_REFERENCIA, STATUS_PENDENTES
from .regras_clinicas import classificar_pendencias, gera_alerta

DB_PATH = Path(__file__).parents[1] / "data" / "synthetic" / "prontuarios.db"
ID_RE = re.compile(r"^P\d{3}$")

ERRO_ID_INVALIDO = {"erro": "id de paciente inválido (formato esperado: P000)"}
ERRO_NAO_ENCONTRADO = {"erro": "paciente não encontrado"}

CAMPOS_PACIENTE = ("id", "idade", "sexo", "diagnostico", "estadio", "status_tratamento",
                   "ultima_consulta")
CAMPOS_EXAME = ("id", "exame", "status", "data_solicitacao", "data_prevista", "data_resultado",
                "resultado", "protocolo_codigo")


def conectar(db_path: Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)


def _checar_paciente(conn: sqlite3.Connection, paciente_id) -> dict | None:
    """Devolve um dict de erro se o id for inválido ou não existir; senão None."""
    if not isinstance(paciente_id, str) or not ID_RE.match(paciente_id):
        return ERRO_ID_INVALIDO
    if conn.execute("SELECT 1 FROM pacientes WHERE id = ?", (paciente_id,)).fetchone() is None:
        return ERRO_NAO_ENCONTRADO
    return None


def obter_paciente(paciente_id: str) -> dict:
    with closing(conectar()) as conn:
        if erro := _checar_paciente(conn, paciente_id):
            return erro
        linha = conn.execute(
            f"SELECT {', '.join(CAMPOS_PACIENTE)} FROM pacientes WHERE id = ?", (paciente_id,)
        ).fetchone()
        protocolos = conn.execute(
            "SELECT protocolo_codigo, motivo FROM paciente_protocolos WHERE paciente_id = ? "
            "ORDER BY protocolo_codigo", (paciente_id,)
        ).fetchall()
    paciente = dict(zip(CAMPOS_PACIENTE, linha))
    paciente["protocolos"] = [{"codigo": c, "motivo": m} for c, m in protocolos]
    return paciente


def listar_exames(paciente_id: str, apenas_pendentes: bool = False) -> dict:
    with closing(conectar()) as conn:
        if erro := _checar_paciente(conn, paciente_id):
            return erro
        sql = f"SELECT {', '.join(CAMPOS_EXAME)} FROM exames WHERE paciente_id = ?"
        params: list = [paciente_id]
        if apenas_pendentes:
            sql += f" AND status IN ({','.join('?' * len(STATUS_PENDENTES))})"
            params += STATUS_PENDENTES
        linhas = conn.execute(sql + " ORDER BY id", params).fetchall()
    return {
        "paciente_id": paciente_id,
        "apenas_pendentes": apenas_pendentes,
        "exames": [dict(zip(CAMPOS_EXAME, l)) for l in linhas],
    }


def verificar_pendencias(paciente_id: str) -> dict:
    with closing(conectar()) as conn:
        if erro := _checar_paciente(conn, paciente_id):
            return erro
        pendencias = classificar_pendencias(paciente_id, DATA_REFERENCIA, conn)
    return {
        "paciente_id": paciente_id,
        "data_referencia": DATA_REFERENCIA,
        "gera_alerta": any(gera_alerta(p["nivel"]) for p in pendencias),
        "pendencias": pendencias,
    }


def pacientes_com_alerta() -> dict:
    """Uma entrada por pendência que gera alerta (ATRASADO ou CRITICO), com o status do exame."""
    with closing(conectar()) as conn:
        ids = [i for (i,) in conn.execute("SELECT id FROM pacientes ORDER BY id")]
        pendencias = [
            {"paciente_id": pid, "exame": p["exame"], "status": p["status"],
             "data_prevista": p["data_prevista"], "nivel": p["nivel"],
             "protocolo_codigo": p["protocolo_codigo"], "justificativa": p["justificativa"]}
            for pid in ids
            for p in classificar_pendencias(pid, DATA_REFERENCIA, conn)
            if gera_alerta(p["nivel"])
        ]
    return {"data_referencia": DATA_REFERENCIA, "pendencias": pendencias}
