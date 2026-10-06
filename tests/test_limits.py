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
