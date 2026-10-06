from insight_engine import config


def test_valores_padrao_quando_nada_configurado():
    assert config.get_gemini_api_key() is None
    assert config.get_gemini_model() == config.DEFAULT_GEMINI_MODEL
    assert config.get_gemini_fallback_model() == config.DEFAULT_GEMINI_FALLBACK_MODEL


def test_le_variaveis_de_ambiente(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "chave-env")
    monkeypatch.setenv("GEMINI_MODEL", "modelo-env")
    assert config.get_gemini_api_key() == "chave-env"
    assert config.get_gemini_model() == "modelo-env"


def test_valor_vazio_usa_o_padrao(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "")
    assert config.get_gemini_model() == config.DEFAULT_GEMINI_MODEL


def test_st_secrets_tem_prioridade_sobre_o_ambiente(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "chave-env")
    monkeypatch.setattr(config, "_from_streamlit_secrets", lambda name: f"{name}-secrets")
    assert config.get_gemini_api_key() == "GEMINI_API_KEY-secrets"
