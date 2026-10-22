"""
Testes das regras de criticidade com um banco em memória.

Uso: poetry run python -m unittest tests.test_regras_clinicas -v
"""

import sqlite3
import unittest

from src.data_prep import SCHEMA_PRONTUARIOS
from src.regras_clinicas import CRITICO, ROTINA, classificar_pendencias


def banco_com_paciente(diagnostico: str) -> sqlite3.Connection:
    """Paciente fictício com uma USG pendente respaldada pelo RAD-001."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PRONTUARIOS)
    conn.execute(
        "INSERT INTO pacientes VALUES ('P900', 'Teste', 50, 'F', ?, NULL, 'seguimento', '2026-10-01')",
        (diagnostico,),
    )
    conn.execute("INSERT INTO paciente_protocolos VALUES ('P900', 'HOSP-2024-RAD-001', 'teste')")
    conn.execute(
        """INSERT INTO exames (paciente_id, exame, data_solicitacao, data_prevista, status,
                               protocolo_codigo)
           VALUES ('P900', 'USG mamária de controle', '2026-10-01', '2026-12-01', 'solicitado',
                   'HOSP-2024-RAD-001')"""
    )
    return conn


class TestRegraSeguimentoBiRads3(unittest.TestCase):
    def test_bi_rads_4_com_usg_do_rad001_e_rotina(self):
        [pendencia] = classificar_pendencias("P900", conn=banco_com_paciente("Nódulo mamário BI-RADS 4B"))
        self.assertEqual(pendencia["nivel"], ROTINA)

    def test_bi_rads_3_com_usg_do_rad001_e_critico(self):
        # Controle positivo: o mesmo exame vira CRITICO quando o diagnóstico é BI-RADS 3.
        [pendencia] = classificar_pendencias(
            "P900", conn=banco_com_paciente("Nódulo BI-RADS 3 — provavelmente benigno"))
        self.assertEqual(pendencia["nivel"], CRITICO)


if __name__ == "__main__":
    unittest.main()
