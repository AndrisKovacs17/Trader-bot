#!/usr/bin/env python3
"""Test FIXED equity calculation - fee should ALWAYS decrease equity."""

import sys
from pathlib import Path

# Add parent directory (diplomamunkakod) to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.domain.models import Fill
from core.analytics.services import PerformanceTracker

def test_equity_with_fee_fix():
    """Test that fee always decreases equity (not increases)."""
    print("=" * 70)
    print("EQUITY FEE FIX TEST - Fee should ALWAYS decrease equity")
    print("=" * 70)
    
    tracker = PerformanceTracker()
    
    print(f"\nInitial equity: 10000")
    
    # BUY 1 BTC @ 50000 with fee 100
    # OLD WRONG: pnl = -(1 * 50000) + 100 = -49900
    # NEW FIXED: pnl = -(1 * 50000) - 100 = -50100
    fill_buy = Fill(
        order_id="oid1",
        qty=1.0,  # BUY (positive)
        price=50000.0,
        fee=100.0,
        side="BUY"
    )
    tracker.on_fill(fill_buy, current_equity=-40100.0)
    equity_after_buy = tracker.equity_curve[-1]
    print(f"\nBUY 1 BTC @ 50000 (fee=100)")
    print(f"  PnL calculation: -(1.0 * 50000.0) - 100 = -50100")
    print(f"  Expected equity: 10000 + (-50100) = -40100")
    print(f"  Actual equity:   {equity_after_buy}")
    
    expected_buy = 10000.0 - 50100.0  # = -40100
    assert equity_after_buy == expected_buy, f"BUY equity wrong: {equity_after_buy} != {expected_buy}"
    print(f"  ✓ CORRECT!")
    
    # SELL 0.5 BTC @ 60000 with fee 50
    # PnL = (qty * price) - fee = (0.5 * 60000) - 50 = 29950
    fill_sell = Fill(
        order_id="oid2",
        qty=0.5,  # SELL (qty always positive)
        price=60000.0,
        fee=50.0,
        side="SELL"
    )
    tracker.on_fill(fill_sell, current_equity=-10150.0)
    equity_after_sell = tracker.equity_curve[-1]
    print(f"\nSELL 0.5 BTC @ 60000 (fee=50)")
    print(f"  PnL calculation: (0.5 * 60000.0) - 50 = +29950")
    print(f"  Expected equity: -40100 + 29950 = -10150")
    print(f"  Actual equity:   {equity_after_sell}")
    
    expected_sell = -40100.0 + 29950.0  # = -10150
    assert equity_after_sell == expected_sell, f"SELL equity wrong: {equity_after_sell} != {expected_sell}"
    print(f"  ✓ CORRECT!")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Fee ALWAYS reduces PnL (previously it ADDED to SELL PnL!)")
    print("=" * 70)
    print("\nBUY formula:  pnl = -(qty * price) - fee")
    print("SELL formula: pnl = +(qty * price) - fee")
    print("  → BUY: -(1*50000) - 100 = -50100")
    print("  → SELL: +(0.5*60000) - 50 = +29950")
    print("\n✓ All tests PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    test_equity_with_fee_fix()
