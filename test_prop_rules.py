# test_prop_rules.py
# Lancer:  pip install -U pytest
#          pytest -q

import pytest
from prop_rules import ApexSpec, ApexState, DailyRiskState, StrategySpec, MarketSpec, daily_risk_allows_new_trade, update_daily_after_trade


def test_trailing_initialization():
    spec = ApexSpec(starting_balance=50_000, trailing_dd=2_500, trailing_mode="strict")
    st = ApexState.init(spec)
    assert st.equity == 50_000
    assert st.peak_equity == 50_000
    assert st.trailing_level == 47_500


def test_trailing_moves_up_with_favorable_unrealized():
    spec = ApexSpec(starting_balance=50_000, trailing_dd=2_500, trailing_mode="strict")
    st = ApexState.init(spec)

    # latent favorable +1000 => peak 51000 => trailing 48500
    st.update_peak_with_favorable_unrealized(spec, favorable_unrealized_usd=1_000)
    assert st.peak_equity == 51_000
    assert st.trailing_level == 48_500


def test_liquidation_when_worst_unrealized_hits_trailing():
    spec = ApexSpec(starting_balance=50_000, trailing_dd=2_500, trailing_mode="strict")
    st = ApexState.init(spec)

    # Peak monte à 51k => trailing 48.5k
    st.update_peak_with_favorable_unrealized(spec, 1_000)

    # worst unrealized = -2600 => equity(50k)+(-2600)=47400 <= 48500 => liquidation
    st.check_liquidation_with_worst_unrealized(-2_600, reason="TEST")
    assert st.liquidated is True
    assert st.liquidation_reason == "TEST"
    assert st.equity == st.trailing_level


def test_daily_risk_rules():
    market = MarketSpec()
    strat = StrategySpec(sl_ticks=15, daily_stop_r=-1.0, max_trades_per_day=2, max_consecutive_losses=2)
    daily = DailyRiskState()

    assert daily_risk_allows_new_trade(daily, strat, market) is True

    # 1ère perte -1R => stop quotidien atteint (<= -1R)
    R_usd = strat.sl_ticks * market.tick_value
    update_daily_after_trade(daily, pnl_usd=-R_usd)
    assert daily.pnl_usd == -R_usd
    assert daily_risk_allows_new_trade(daily, strat, market) is False


def test_stop_after_two_losses_even_if_pnl_not_below_1R():
    market = MarketSpec()
    strat = StrategySpec(sl_ticks=15, daily_stop_r=-10.0, max_trades_per_day=10, max_consecutive_losses=2)
    daily = DailyRiskState()

    update_daily_after_trade(daily, pnl_usd=-1.0)
    assert daily_risk_allows_new_trade(daily, strat, market) is True
    update_daily_after_trade(daily, pnl_usd=-1.0)
    assert daily_risk_allows_new_trade(daily, strat, market) is False
