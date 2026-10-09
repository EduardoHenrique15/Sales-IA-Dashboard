import numpy as np
import pandas as pd
import pytest

from insight_engine.analytics.anomalies import daily_series, decompose, detect_anomalies, monthly_seasonality
from insight_engine.data.sales import PLANTED_ANOMALIES


def test_detector_encontra_as_anomalias_plantadas_com_poucos_alarmes(sales_df):
    """Validação contra gabarito: a base de exemplo tem anomalias em datas conhecidas."""
    found = detect_anomalies(daily_series(sales_df, "orders"))
    planted = {pd.Timestamp(day): kind for day, kind, _ in PLANTED_ANOMALIES}

    for day, kind in planted.items():
        assert day in found.index, f"anomalia plantada em {day:%d/%m/%Y} não detectada"
        assert found.loc[day, "kind"] == kind
    false_alarm_rate = (len(found) - len(planted)) / len(sales_df["date"].dt.normalize().unique())
    assert false_alarm_rate < 0.01  # menos de 1% dos dias


@pytest.mark.parametrize("seed", range(5))
def test_serie_regular_nao_tem_anomalias(seed):
    index = pd.date_range("2025-01-01", periods=120, freq="D")
    rng = np.random.default_rng(seed)
    series = pd.Series(100 + np.where(index.dayofweek >= 5, -20, 0) + rng.normal(0, 2, 120), index=index)
    assert detect_anomalies(series).empty


def test_desvio_pequeno_nao_e_relevante_mesmo_se_raro():
    index = pd.date_range("2025-01-01", periods=120, freq="D")
    series = pd.Series(100.0, index=index) + np.tile([0.5, -0.5], 60)
    series.iloc[60] = 110  # +10%: estatisticamente extremo nesta série, mas abaixo da materialidade
    assert detect_anomalies(series).empty
    assert not detect_anomalies(series, min_deviation_pct=5).empty


def test_pico_isolado_e_detectado_e_classificado():
    index = pd.date_range("2025-01-01", periods=120, freq="D")
    rng = np.random.default_rng(2)
    series = pd.Series(100 + rng.normal(0, 3, 120), index=index)
    series.iloc[60] = 400
    found = detect_anomalies(series)
    assert found.index[0] == index[60]
    assert found.iloc[0]["kind"] == "pico"
    assert found.iloc[0]["deviation_pct"] > 200


def test_historico_curto_demais():
    with pytest.raises(ValueError, match="pelo menos"):
        detect_anomalies(pd.Series([1.0] * 10, index=pd.date_range("2025-01-01", periods=10)))


def test_serie_de_pedidos_conta_pedidos_por_dia():
    df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-03"]), "revenue": [1.0, 2.0, 3.0]})
    assert daily_series(df, "orders").tolist() == [2.0, 0.0, 1.0]
    assert daily_series(df, "revenue").tolist() == [3.0, 0.0, 3.0]


def test_fim_de_semana_vende_menos_na_base_de_exemplo(sales_df):
    profile = decompose(daily_series(sales_df)).weekday_profile
    assert profile["Sáb"] < 0 and profile["Dom"] < 0
    assert profile["Seg"] > 0


def test_dezembro_e_o_pico_sazonal_da_base_de_exemplo(sales_df):
    index = monthly_seasonality(daily_series(sales_df))
    assert index.idxmax() == "Dez"
    assert index["Dez"] > 1.3


def test_serie_vai_ate_o_fim_da_base():
    # um segmento que parou de vender tem dias zerados até o fim da base
    df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "revenue": [1.0, 2.0]})
    assert daily_series(df, end=pd.Timestamp("2025-01-04 15:00")).tolist() == [1.0, 2.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="Não há vendas"):
        daily_series(df.iloc[0:0])


def test_sazonalidade_mensal_ignora_ano_incompleto():
    index = pd.date_range("2024-01-01", "2025-03-31", freq="D")
    # 2024 completo e plano; 2025 só até março, com um janeiro atípico que distorceria o índice
    series = pd.Series(np.where((index.year == 2025) & (index.month == 1), 500.0, 100.0), index=index)
    assert monthly_seasonality(series).tolist() == pytest.approx([1.0] * 12)
