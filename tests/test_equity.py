#!/usr/bin/env python3
"""Test equity calculation with proper BUY/SELL handling."""

import sys
from pathlib import Path

# Add parent directory (diplomamunkakod) to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.domain.models import Fill
from core.analytics.services import PerformanceTracker

def test_equity_calculation():
    """Test PerformanceTracker.on_fill() with corrected PnL calculation."""
    print("=" * 60)
    print("Equity calculation test (BUY/SELL PnL)")
    print("=" * 60)
    
    tracker = PerformanceTracker()
    
    # Initial equity: 10000
    print(f"\nInitial equity: 10000")
    
    # BUY 1 BTC @ 50000 with fee 100
    # NEW FIXED: PnL = -(qty * price) - abs(fee) = -(1 * 50000) - 100 = -50100
    # equity = 10000 + (-50100) = -40100
    fill_buy = Fill(
        order_id="oid1",
        qty=1.0,  # BUY (positive)
        price=50000.0,
        fee=100.0,
        side="BUY"
    )
    tracker.on_fill(fill_buy, current_equity=-40100.0)
    print(f"After BUY 1 BTC @ 50000: equity = {tracker.equity_curve[-1]}")
    
    # Expected: 10000 + (-(1 * 50000) - 100) = 10000 - 50100 = -40100
    assert tracker.equity_curve[-1] == -40100, f"BUY equity wrong: {tracker.equity_curve[-1]}"
    print("✓ BUY equity calculated correctly")
    
    # SELL 0.5 BTC @ 60000 with fee 50
    # NEW: PnL = (qty * price) - fee = (0.5 * 60000) - 50 = 29950
    # equity = -40100 + 29950 = -10150
    fill_sell = Fill(
        order_id="oid2",
        qty=0.5,  # SELL (qty always positive)
        price=60000.0,
        fee=50.0,
        side="SELL"
    )
    tracker.on_fill(fill_sell, current_equity=-10150.0)
    print(f"After SELL 0.5 BTC @ 60000: equity = {tracker.equity_curve[-1]}")
    
    # Expected: -40100 + ((0.5 * 60000) - 50) = -10150
    assert tracker.equity_curve[-1] == -10150, f"SELL equity wrong: {tracker.equity_curve[-1]}"
    print("✓ SELL equity calculated correctly")
    
    # Snapshot should reflect current equity and cash
    snap = tracker.snapshot()
    print(f"\nSnapshot equity: {snap.equity}")
    print(f"Snapshot positions: {snap.positions}")
    assert snap.equity == -10150, f"Snapshot equity wrong: {snap.equity}"
    assert "cash" in snap.positions, "Snapshot missing cash"
    
    print("\n" + "=" * 60)
    print("✓ All equity calculation tests PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_equity_calculation()
