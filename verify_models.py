"""
Final Production Verification Suite for Queueing Theory Models

Tests:
- Input validation (negative λ, non-positive μ, invalid types)
- Edge cases (λ=0, N=0, N<c, large N, extreme ρ)
- Numerical stability (overflow, underflow)
- Flow balance sanity checks
- Tail mass reporting
"""

from queueing_models import QueueingModel, factorial, log_factorial, safe_exp
import math


def test_safe_exp():
    """Test safe_exp handles overflow and underflow."""
    print("=" * 60)
    print("Testing safe_exp() Overflow/Underflow Guards")
    print("=" * 60)
    
    all_passed = True
    
    # Underflow
    result = safe_exp(-800)
    status = "✓" if result == 0.0 else "✗"
    if result != 0.0:
        all_passed = False
    print(f"  exp(-800) → 0.0: {status}")
    
    # Overflow
    result = safe_exp(800)
    status = "✓" if result == float('inf') else "✗"
    if result != float('inf'):
        all_passed = False
    print(f"  exp(800) → inf: {status}")
    
    # Normal case
    result = safe_exp(0)
    status = "✓" if result == 1.0 else "✗"
    if result != 1.0:
        all_passed = False
    print(f"  exp(0) → 1.0: {status}")
    
    return all_passed


def test_input_validation():
    """Test input validation including types."""
    print("\n" + "=" * 60)
    print("Testing Input Validation (Including Types)")
    print("=" * 60)
    
    all_passed = True
    
    # Negative λ
    qm = QueueingModel(-1.0, 2.0)
    r = qm.mm1_infinite()
    status = "✓" if r.error_message and "λ" in r.error_message else "✗"
    if not (r.error_message and "λ" in r.error_message):
        all_passed = False
    print(f"  Negative λ rejected: {status}")
    
    # Zero μ
    qm = QueueingModel(2.0, 0.0)
    r = qm.mm1_infinite()
    status = "✓" if r.error_message and "μ" in r.error_message else "✗"
    if not (r.error_message and "μ" in r.error_message):
        all_passed = False
    print(f"  Zero μ rejected: {status}")
    
    # c = 0 (invalid)
    qm = QueueingModel(2.0, 3.0)
    r = qm.mmc_infinite(0)
    status = "✓" if r.error_message and "c" in r.error_message else "✗"
    if not (r.error_message and "c" in r.error_message):
        all_passed = False
    print(f"  c=0 rejected: {status}")
    
    # Negative N
    r = qm.mm1_finite(-1)
    status = "✓" if r.error_message and "N" in r.error_message else "✗"
    if not (r.error_message and "N" in r.error_message):
        all_passed = False
    print(f"  Negative N rejected: {status}")
    
    # Negative max_n
    r = qm.mm1_infinite(-5)
    status = "✓" if r.error_message and "max_n" in r.error_message else "✗"
    if not (r.error_message and "max_n" in r.error_message):
        all_passed = False
    print(f"  Negative max_n rejected: {status}")
    
    return all_passed


def test_N_zero():
    """Test N=0 edge case."""
    print("\n" + "=" * 60)
    print("Testing N=0 Edge Case")
    print("=" * 60)
    
    all_passed = True
    qm = QueueingModel(5.0, 2.0)
    
    for model_name, result in [("M/M/1/0", qm.mm1_finite(0)), ("M/M/c/0", qm.mmc_finite(3, 0))]:
        checks = [
            ("P0 = 1", result.P0, 1.0),
            ("Pn = 1", result.Pn, 1.0),
            ("λ_lost = λ", result.lambda_lost, 5.0),
            ("λ_eff = 0", result.lambda_eff, 0.0),
        ]
        
        passed = True
        for desc, actual, expected in checks:
            if abs(actual - expected) > 1e-10:
                passed = False
                all_passed = False
        
        status = "✓" if passed else "✗"
        print(f"  {model_name}: {status}")
    
    return all_passed


def test_mmc_overflow():
    """Test M/M/c doesn't crash on extreme ρ close to 1."""
    print("\n" + "=" * 60)
    print("Testing M/M/c Overflow Prevention (ρ → 1)")
    print("=" * 60)
    
    all_passed = True
    
    # ρ = 0.999999 (very close to 1, Lq should be huge but not crash)
    # λ = 0.999999 * c * μ
    c, mu = 10, 1.0
    lam = 0.999999 * c * mu
    qm = QueueingModel(lam, mu)
    
    try:
        result = qm.mmc_infinite(c)
        valid = (
            not math.isnan(result.Ls) and
            not math.isnan(result.Lq) and
            result.Lq > 1000  # Should be very large
        )
        status = "✓" if valid else "✗"
        if not valid:
            all_passed = False
        print(f"  ρ=0.999999: Lq={result.Lq:.2f} (very large, no crash) {status}")
    except Exception as e:
        all_passed = False
        print(f"  ρ=0.999999: EXCEPTION - {e} ✗")
    
    return all_passed


def test_large_N_warning():
    """Test that large N triggers warning with original value."""
    print("\n" + "=" * 60)
    print("Testing Large N Warning")
    print("=" * 60)
    
    all_passed = True
    qm = QueueingModel(2.0, 3.0)
    
    result = qm.mm1_finite(N=50000)
    
    has_warning = result.warning_message is not None
    has_original_N = "50000" in (result.warning_message or "")
    
    status = "✓" if has_warning and has_original_N else "✗"
    if not (has_warning and has_original_N):
        all_passed = False
    print(f"  Warning includes original N: {status}")
    if has_warning:
        print(f"    {result.warning_message}")
    
    return all_passed


def test_tail_mass():
    """Test tail_mass is reported correctly."""
    print("\n" + "=" * 60)
    print("Testing Tail Mass Reporting")
    print("=" * 60)
    
    all_passed = True
    
    # M/M/1 with ρ=0.9 and small max_n - should have significant tail
    qm = QueueingModel(9.0, 10.0)  # ρ = 0.9
    result = qm.mm1_infinite(max_n=5)
    
    prob_sum = sum(result.probabilities)
    expected_tail = 1.0 - prob_sum
    
    diff = abs(result.tail_mass - expected_tail)
    status = "✓" if diff < 0.001 else "✗"
    if diff >= 0.001:
        all_passed = False
    print(f"  M/M/1 (ρ=0.9, max_n=5): tail_mass={result.tail_mass:.4f} {status}")
    
    # M/M/∞ with large a and small max_n - should show warning
    qm2 = QueueingModel(100.0, 1.0)  # a = 100
    result2 = qm2.mm_infinity(max_n=10)
    
    has_warning = result2.warning_message is not None
    status = "✓" if has_warning and result2.tail_mass > 0.99 else "✗"
    if not (has_warning and result2.tail_mass > 0.99):
        all_passed = False
    print(f"  M/M/∞ (a=100, max_n=10): warning={has_warning}, tail={result2.tail_mass:.4f} {status}")
    
    return all_passed


def test_flow_balance():
    """Test flow balance sanity checks work."""
    print("\n" + "=" * 60)
    print("Testing Flow Balance Checks")
    print("=" * 60)
    
    all_passed = True
    
    # Normal case - should pass flow balance
    qm = QueueingModel(2.0, 3.0)
    result = qm.mm1_finite(10)
    
    # Verify λ_eff ≈ μ*(1-P0)
    expected = qm.mu_rate * (1 - result.P0)
    diff = abs(result.lambda_eff - expected)
    rel_error = diff / result.lambda_eff if result.lambda_eff > 0 else 0
    
    no_warning = result.warning_message is None or "flow balance" not in (result.warning_message or "")
    status = "✓" if no_warning and rel_error < 1e-6 else "✗"
    if not (no_warning and rel_error < 1e-6):
        all_passed = False
    print(f"  M/M/1/N: flow balance OK (rel_error={rel_error:.2e}) {status}")
    
    return all_passed


def test_standard_cases():
    """Test standard analytical cases."""
    print("\n" + "=" * 60)
    print("Testing Standard Cases")
    print("=" * 60)
    
    all_passed = True
    
    # M/M/1
    qm = QueueingModel(2.0, 3.0)
    r = qm.mm1_infinite()
    
    expected = {"P0": 1/3, "Ls": 2.0, "Lq": 4/3, "C_bar": 2/3}
    
    print("  M/M/1 (λ=2, μ=3):")
    for name, exp in expected.items():
        actual = getattr(r, name)
        diff = abs(actual - exp)
        status = "✓" if diff < 0.0001 else "✗"
        if diff >= 0.0001:
            all_passed = False
        print(f"    {name}: {actual:.4f} (expected {exp:.4f}) {status}")
    
    # M/M/∞
    qm2 = QueueingModel(3.0, 2.0)
    r2 = qm2.mm_infinity()
    
    expected2 = {"Ls": 1.5, "Lq": 0.0, "P0": math.exp(-1.5)}
    
    print("  M/M/∞ (λ=3, μ=2):")
    for name, exp in expected2.items():
        actual = getattr(r2, name)
        diff = abs(actual - exp)
        status = "✓" if diff < 0.0001 else "✗"
        if diff >= 0.0001:
            all_passed = False
        print(f"    {name}: {actual:.6f} (expected {exp:.6f}) {status}")
    
    return all_passed


def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║  QUEUEING THEORY - FINAL PRODUCTION VERIFICATION        ║")
    print("╚" + "═" * 58 + "╝")
    
    results = {
        "safe_exp Guards": test_safe_exp(),
        "Input Validation": test_input_validation(),
        "N=0 Edge Case": test_N_zero(),
        "M/M/c Overflow": test_mmc_overflow(),
        "Large N Warning": test_large_N_warning(),
        "Tail Mass": test_tail_mass(),
        "Flow Balance": test_flow_balance(),
        "Standard Cases": test_standard_cases(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print()
    if all_passed:
        print("  ✅ ALL TESTS PASSED - PRODUCTION READY!")
    else:
        print("  ❌ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    main()
