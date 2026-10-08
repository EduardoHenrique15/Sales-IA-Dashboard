from insight_engine.ai.prompts import CHAT_SYSTEM, REPORT_SYSTEM, build_report_prompt


def test_prompt_leva_os_fatos_e_as_regras():
    prompt = build_report_prompt("- Receita total: R$ 10,00", "Vendas")
    assert "Relatório Executivo de Vendas" in prompt
    assert "- Receita total: R$ 10,00" in prompt
    for field in ("headline", "highlights", "risks", "actions"):
        assert field in prompt


def test_instrucoes_proibem_numeros_inventados():
    assert "SOMENTE números que aparecem no bloco de DADOS" in REPORT_SYSTEM
    assert "nunca estime nem invente valores" in CHAT_SYSTEM
    assert "só responde sobre os dados" in CHAT_SYSTEM  # recusa perguntas fora do assunto
