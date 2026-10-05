from unittest import mock

import pytest
import requests

from insight_engine.data import crypto
from insight_engine.data.crypto import CryptoDataError, _parse_market_chart
from tests.helpers import DAY_MS, market_chart


def _response(status=200, payload=None, json_error=False):
    resp = mock.Mock(status_code=status)
    resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}") if status >= 400 else None
    resp.json.side_effect = ValueError("JSON inválido") if json_error else None
    resp.json.return_value = payload
    return resp


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(crypto.time, "sleep", lambda s: None)


class TestParseMarketChart:
    def test_converte_em_uma_linha_por_dia(self):
        df = _parse_market_chart(market_chart([10, 20, 30], [1, 2, 3]))
        assert list(df.columns) == ["date", "price", "volume"]
        assert df["price"].tolist() == [10, 20, 30]
        assert df["volume"].tolist() == [1, 2, 3]

    def test_une_preco_e_volume_pelo_timestamp_e_nao_pela_posicao(self):
        payload = market_chart([10, 20], [1, 2])
        payload["total_volumes"].reverse()
        df = _parse_market_chart(payload)
        assert df["volume"].tolist() == [1, 2]

    def test_ponto_extra_do_dia_atual_substitui_o_fechamento(self):
        payload = market_chart([10, 20])
        last_ts = payload["prices"][-1][0]
        payload["prices"].append([last_ts + 3_600_000, 21])  # 1h depois, mesmo dia
        payload["total_volumes"].append([last_ts + 3_600_000, 999])

        df = _parse_market_chart(payload)
        assert len(df) == 2
        assert df["price"].iloc[-1] == 21
        assert df["volume"].iloc[-1] == 999

    def test_volume_ausente_vira_zero(self):
        payload = {"prices": [[20_000 * DAY_MS, 10]], "total_volumes": []}
        assert _parse_market_chart(payload)["volume"].tolist() == [0]

    @pytest.mark.parametrize("payload", [{}, {"prices": []}, {"prices": [[0, None]]}])
    def test_resposta_vazia(self, payload):
        with pytest.raises(CryptoDataError) as exc:
            _parse_market_chart(payload)
        assert exc.value.kind == "empty_response"

    def test_preco_invalido(self):
        with pytest.raises(CryptoDataError) as exc:
            _parse_market_chart(market_chart([10, -5]))
        assert exc.value.kind == "invalid_response"


class TestLoadCryptoData:
    def test_sucesso(self, monkeypatch):
        get = mock.Mock(return_value=_response(payload=market_chart([1, 2, 3])))
        monkeypatch.setattr(crypto.requests, "get", get)

        df = crypto.load_crypto_data("Ethereum (ETH)", days=30)

        assert len(df) == 3
        url = get.call_args.args[0]
        assert "/coins/ethereum/market_chart" in url
        assert get.call_args.kwargs["params"]["days"] == 30
        assert get.call_args.kwargs["headers"] == {}

    def test_envia_a_chave_demo_quando_configurada(self, monkeypatch):
        monkeypatch.setenv("COINGECKO_API_KEY", "demo-123")
        get = mock.Mock(return_value=_response(payload=market_chart([1])))
        monkeypatch.setattr(crypto.requests, "get", get)

        crypto.load_crypto_data()
        assert get.call_args.kwargs["headers"] == {"x-cg-demo-api-key": "demo-123"}

    def test_rate_limit_tenta_de_novo_e_depois_desiste(self, monkeypatch, no_sleep):
        get = mock.Mock(return_value=_response(status=429))
        monkeypatch.setattr(crypto.requests, "get", get)

        with pytest.raises(CryptoDataError) as exc:
            crypto.load_crypto_data(max_retries=3)
        assert exc.value.kind == "rate_limit"
        assert get.call_count == 3

    def test_rate_limit_seguido_de_sucesso(self, monkeypatch, no_sleep):
        get = mock.Mock(side_effect=[_response(status=429), _response(payload=market_chart([1, 2]))])
        monkeypatch.setattr(crypto.requests, "get", get)
        assert len(crypto.load_crypto_data()) == 2

    def test_falha_de_rede(self, monkeypatch, no_sleep):
        get = mock.Mock(side_effect=requests.ConnectionError("sem conexão"))
        monkeypatch.setattr(crypto.requests, "get", get)

        with pytest.raises(CryptoDataError) as exc:
            crypto.load_crypto_data(max_retries=2)
        assert exc.value.kind == "network_error"
        assert get.call_count == 2

    def test_erro_http_e_tratado_como_falha_de_rede(self, monkeypatch, no_sleep):
        monkeypatch.setattr(crypto.requests, "get", mock.Mock(return_value=_response(status=500)))
        with pytest.raises(CryptoDataError) as exc:
            crypto.load_crypto_data(max_retries=1)
        assert exc.value.kind == "network_error"

    def test_json_malformado(self, monkeypatch):
        monkeypatch.setattr(crypto.requests, "get", mock.Mock(return_value=_response(json_error=True)))
        with pytest.raises(CryptoDataError) as exc:
            crypto.load_crypto_data()
        assert exc.value.kind == "invalid_response"
