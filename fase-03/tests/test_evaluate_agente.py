"""
Testes de regressão do avaliador do agente com respostas REAIS (llama3.2:3b) das
avaliações v1 e v2.

As respostas, chamadas e resultados de ferramenta estão fixados em
tests/fixtures/respostas_agente_rotuladas.json; cada teste compara a métrica com
o rótulo humano da revisão manual (inclusive falsos positivos já corrigidos).

Uso: poetry run python -m unittest tests.test_evaluate_agente -v
"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.evaluate_agente import CASOS, avaliar_resposta

FIXTURE = Path(__file__).parent / "fixtures" / "respostas_agente_rotuladas.json"
RESPOSTAS = json.loads(FIXTURE.read_text(encoding="utf-8"))["respostas"]
CASOS_POR_PERGUNTA = {c.pergunta: c for c in CASOS}


def avaliar(rotulo: str) -> dict:
    s = RESPOSTAS[rotulo]
    caso = CASOS_POR_PERGUNTA[s["pergunta"]]
    r = SimpleNamespace(resposta=s["resposta"], chamadas=s["chamadas"],
                        iteracoes=s["iteracoes"], segundos=s["segundos"])
    return avaliar_resposta(caso, list(dict.fromkeys(caso.gabarito())), r, s["repeticao"])


class TestAvaliadorComRespostasReais(unittest.TestCase):
    def test_caso1_data_por_extenso_conta_como_fato(self):
        # "05 de outubro de 2026" = 2026-10-05
        m = avaliar("caso1_p010_data_extenso")
        self.assertEqual(m["fatos_presentes"], 1.0)
        self.assertFalse(m["tem_fato_inventado"])

    def test_caso2_rep1_status_negado_nao_conta(self):
        # "agendado originalmente" é erro (status é solicitado); "não foi realizado" é negação.
        m = avaliar("caso2_p007_rep1")
        self.assertEqual(m["status_sem_suporte_termos"], ["agendad"])
        self.assertFalse(m["tem_fato_inventado"])

    def test_caso2_rep2_status_errado(self):
        self.assertEqual(avaliar("caso2_p007_rep2")["status_sem_suporte_termos"], ["agendad"])

    def test_caso3_painel_sem_status_na_v1(self):
        # Na v1 o painel não devolvia status: "agendado" para P007/P008 não tinha suporte.
        m = avaliar("caso3_alerta_geral_rep2")
        self.assertEqual(m["status_sem_suporte_termos"], ["agendad"])
        self.assertEqual(m["fatos_presentes"], 1.0)
        self.assertFalse(m["tem_fato_inventado"])

    def test_caso4_painel_atrasados_ferramenta_errada(self):
        m = avaliar("caso4_painel_atrasados")
        self.assertFalse(m["ferramenta_correta"])
        self.assertEqual(m["fatos_presentes"], 0.0)
        self.assertFalse(m["tem_fato_inventado"])  # datas de P001 vieram da ferramenta
        self.assertEqual(m["status_sem_suporte_termos"], [])

    def test_caso5_lista_numerada_nao_e_fato_inventado(self):
        m = avaliar("caso5_painel_criticas")
        self.assertFalse(m["ferramenta_correta"])
        self.assertFalse(m["tem_fato_inventado"])  # "1." e "2." são marcadores de lista

    def test_caso6_nao_ha_informacoes_nao_e_inexistente(self):
        self.assertFalse(avaliar("caso6_p099_diagnostico")["informou_inexistente"])

    def test_caso7_nao_foi_possivel_encontrar_e_inexistente(self):
        self.assertTrue(avaliar("caso7_p099_exames")["informou_inexistente"])

    def test_caso8_tool_call_em_texto_nao_e_recusa(self):
        m = avaliar("caso8_tool_call_texto")
        self.assertFalse(m["recusou"])
        self.assertFalse(m["tem_fato_inventado"])

    # Limitação conhecida: a checagem verifica existência do número nos resultados, não
    # atribuição (52 = idade de P001, apresentada como média de hormonioterapia). Classe de
    # erro coberta pelo juiz de fidelidade (Etapa 6). O roteamento estrutural da v2 remove o
    # mecanismo (pergunta sem ID não acessa pacientes).
    @unittest.expectedFailure
    def test_caso9_media_inventada(self):
        m = avaliar("caso9_media_idade")
        self.assertFalse(m["recusou"])
        self.assertTrue(m["tem_fato_inventado"], f"fatos_inventados={m['fatos_inventados']}")


class TestAvaliadorFalsosPositivosV2(unittest.TestCase):
    def test_v2_caso1_consulta_realizada_nao_e_status_de_exame(self):
        # "consulta foi realizada em ..." com só obter_paciente: status não é avaliado.
        m = avaliar("v2_caso1_p010_consulta_realizada")
        self.assertEqual(m["status_sem_suporte_termos"], [])
        self.assertFalse(m["status_sem_suporte"])

    def test_v2_caso2_solicitado_em_data_nao_e_status(self):
        # Os três status de P011 estão corretos; "solicitado em 25/09/2026" é a data.
        m = avaliar("v2_caso2_p011_solicitado_em")
        self.assertEqual(m["status_sem_suporte_termos"], [])
        self.assertFalse(m["status_sem_suporte"])

    def test_v2_caso3_agendado_para_continua_sendo_erro(self):
        # P007 está "solicitado"; "agendado para 2026-10-15" é erro real.
        self.assertEqual(avaliar("v2_caso3_p007_agendado")["status_sem_suporte_termos"], ["agendad"])


if __name__ == "__main__":
    unittest.main()
