from insight_engine.ai import limits


def test_janela_deslizante():
    limiter = limits.SlidingWindowLimiter(max_calls=2, window_seconds=60)
    assert limiter.try_acquire(now=0)
    assert limiter.try_acquire(now=10)
    assert not limiter.try_acquire(now=20)  # 3ª chamada dentro de 60 s
    assert limiter.try_acquire(now=61)  # a 1ª saiu da janela


def test_limites_configuraveis(monkeypatch):
    monkeypatch.setenv("SERVER_KEY_CALLS_PER_HOUR", "1")
    limiter = limits.session_limiter()
    assert limiter.try_acquire() and not limiter.try_acquire()
    assert limits.global_limiter().max_calls == limits.DEFAULT_GLOBAL_PER_DAY


def test_cache_limitado_descarta_o_mais_antigo():
    cache = limits.BoundedCache(max_items=2)
    cache["a"], cache["b"] = 1, 2
    _ = cache["a"]  # "a" passa a ser o mais recente
    cache["c"] = 3
    assert list(cache) == ["a", "c"]


def test_limite_global_nao_gasta_a_vaga_do_visitante(monkeypatch):
    from types import SimpleNamespace
    from unittest import mock

    from insight_engine.ui import ai_access

    session = limits.SlidingWindowLimiter(max_calls=1, window_seconds=3600)
    shared = limits.SlidingWindowLimiter(max_calls=0, window_seconds=86400)  # limite global já atingido
    fake_st = SimpleNamespace(session_state={ai_access._SESSION_LIMITER_STATE: session})
    monkeypatch.setattr(ai_access, "get_gemini_api_key", lambda: "chave-do-servidor")
    with (
        mock.patch.object(ai_access, "st", fake_st),
        mock.patch.object(ai_access, "create_provider", return_value=(object(), None)),
        mock.patch.object(ai_access, "_global_limiter", return_value=shared),
    ):
        access = ai_access.ai_access()
    assert access.allow_call is not None
    assert not access.allow_call()
    assert session.try_acquire()  # a vaga da sessão continua livre
