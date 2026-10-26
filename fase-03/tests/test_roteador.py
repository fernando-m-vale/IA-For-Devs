"""
Testes do roteador do grafo (determinístico, sem LLM).

Uso: poetry run python -m unittest tests.test_roteador -v
"""

import unittest

from src.grafo import ID_RE, escolher_rota

CASOS = [
    # Com ID: sempre agente_b, mesmo com vocabulário de painel ou estatística
    ("Qual o diagnóstico do paciente P004?", "agente_b"),
    ("O paciente P007 tem exames atrasados?", "agente_b"),
    ("Quantos exames o paciente P011 tem?", "agente_b"),
    ("Compare as pendências de P002 e P012.", "agente_b"),
    ("O paciente P099 existe?", "agente_b"),
    # Painel
    ("Quais pacientes estão com exames atrasados?", "painel"),
    ("Liste as pendências críticas do hospital.", "painel"),
    ("Há algum alerta ativo hoje?", "painel"),
    ("Há seguimento vencido em algum paciente?", "painel"),
    ("Existe alguma pendência de exame?", "painel"),
    ("Quais exames estão em situação crítica?", "painel"),
    # Ambíguos painel + estatística: painel tem precedência
    ("Quantos pacientes têm exames atrasados?", "painel"),
    ("Qual o total de pendências críticas?", "painel"),
    ("Quantas pendências geram alerta?", "painel"),
    ("Qual a proporção de exames atrasados?", "painel"),
    # Estatística
    ("Quantos pacientes há por status de tratamento?", "agente_c"),
    ("Qual a idade média dos pacientes com carcinoma?", "agente_c"),
    ("Quantos exames foram cancelados?", "agente_c"),
    ("Qual o percentual de pacientes do sexo feminino?", "agente_c"),
    ("Quantos exames pendentes existem no total?", "agente_c"),   # "pendentes" não é "pendência"
    ("Qual a média de idade em hormonioterapia?", "agente_c"),
    # Protocolos (RAG)
    ("Como manejar neutropenia febril em paciente oncológico?", "rag"),
    ("O que é o sistema BI-RADS?", "rag"),
    ("Quando indicar quimioterapia neoadjuvante?", "rag"),
    ("Como comunicar um diagnóstico de câncer ao paciente?", "rag"),
    ("Quais os critérios de internação em emergência oncológica?", "rag"),  # "critérios" não é "crítico"
    ("O protocolo de dor cobre totalmente os opioides fracos?", "rag"),     # "totalmente" não é "total"
]


class TestRoteador(unittest.TestCase):
    def test_rotas(self):
        for pergunta, esperada in CASOS:
            with self.subTest(pergunta=pergunta):
                rota, motivo = escolher_rota(pergunta, sorted(set(ID_RE.findall(pergunta))))
                self.assertEqual(rota, esperada, motivo)

    def test_motivo_registra_ambiguidade(self):
        _, motivo = escolher_rota("Quantos pacientes têm exames atrasados?", [])
        self.assertIn("painel tem precedência", motivo)


if __name__ == "__main__":
    unittest.main()
