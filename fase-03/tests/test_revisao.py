"""
Testes do modo revisão do grafo, com a chamada ao LLM (gerar_conduta) mockada.

Uso: poetry run python -m unittest tests.test_revisao -v
"""

import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage, SystemMessage

from src import grafo
from src.data_augmentation import SYSTEM_PROMPT_RAG
from src.regras_clinicas import classificar_pendencias, gera_alerta

PACIENTES = [f"P{i:03d}" for i in range(1, 13)]
CONDUTA_MOCK = "Conduta de teste.\n\nFonte: Protocolo Interno HOSP-2024-ONCO-001"


class TestRevisao(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("src.grafo.gerar_conduta", return_value=CONDUTA_MOCK)
        self.mock = self.patcher.start()
        self.grafo = grafo.construir_grafo()

    def tearDown(self):
        self.patcher.stop()

    def test_alertas_batem_com_classificar_pendencias(self):
        com_alerta = []
        for pid in PACIENTES:
            with self.subTest(paciente=pid):
                estado, config = grafo.iniciar_revisao(self.grafo, pid)
                self.assertTrue(grafo.aguardando_aprovacao(self.grafo, config))
                esperado = {(p["exame_id"], p["nivel"]) for p in classificar_pendencias(pid)
                            if gera_alerta(p["nivel"])}
                self.assertEqual({(a["exame_id"], a["nivel"]) for a in estado["alertas"]}, esperado)
                self.assertTrue(estado["resposta"].endswith(grafo.RODAPE_REVISAO))
                if esperado:
                    com_alerta.append(pid)
        self.assertEqual(com_alerta, ["P002", "P004", "P007", "P008", "P012"])

    def test_conduta_usa_protocolos_vinculados_no_formato_do_treino(self):
        estado, _ = grafo.iniciar_revisao(self.grafo, "P008")
        [mensagens] = self.mock.call_args.args
        self.assertIsInstance(mensagens[0], SystemMessage)
        self.assertEqual(mensagens[0].content, SYSTEM_PROMPT_RAG)
        self.assertIsInstance(mensagens[1], HumanMessage)
        self.assertTrue(mensagens[1].content.startswith("Contexto (protocolos internos recuperados):"))
        for p in estado["protocolos"]:
            self.assertIn(p["codigo"], mensagens[1].content)
        self.assertEqual({p["codigo"] for p in estado["protocolos"]},
                         {"HOSP-2024-ONCO-003", "HOSP-2024-ONCO-004"})

    def test_paciente_inexistente_encerra_cedo(self):
        estado, config = grafo.iniciar_revisao(self.grafo, "P099")
        self.assertFalse(grafo.aguardando_aprovacao(self.grafo, config))
        self.assertTrue(estado["resposta"].startswith("O paciente P099 não foi encontrado"))
        self.assertTrue(estado["resposta"].endswith(grafo.RODAPE_REVISAO))
        self.assertEqual([e["no"] for e in estado["auditoria"]], ["carregar_paciente"])
        self.mock.assert_not_called()

    def test_interrupt_pausa_e_resume_registra_decisao(self):
        estado, config = grafo.iniciar_revisao(self.grafo, "P007")
        self.assertNotIn("aprovacao", estado)
        self.assertEqual(estado["auditoria"][-1]["no"], "gerar_alertas")

        final = grafo.decidir(self.grafo, config, "aprovado", "conferido")
        self.assertEqual(final["aprovacao"], {"decisao": "aprovado", "comentario": "conferido"})
        self.assertEqual([e["no"] for e in final["auditoria"]][-2:],
                         ["aprovacao_medica", "registrar_decisao"])
        self.assertIn("aprovado", final["auditoria"][-1]["resumo"])
        self.assertIn("conferido", final["auditoria"][-1]["resumo"])
        self.assertFalse(grafo.aguardando_aprovacao(self.grafo, config))

    def test_rejeicao_tambem_e_registrada(self):
        _, config = grafo.iniciar_revisao(self.grafo, "P002")
        final = grafo.decidir(self.grafo, config, "rejeitado", "aguardar laudo")
        self.assertEqual(final["aprovacao"]["decisao"], "rejeitado")

    def test_decisao_invalida_falha(self):
        _, config = grafo.iniciar_revisao(self.grafo, "P004")
        with self.assertRaises(ValueError):
            grafo.decidir(self.grafo, config, "talvez")


if __name__ == "__main__":
    unittest.main()
