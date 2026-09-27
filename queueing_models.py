"""
Queueing Theory Models Module (Production-Ready v2)
Implements Kendall notation queueing models for Decision Support System.

Production features:
- Input validation (rejects negative λ, non-positive μ, invalid types)
- Overflow AND underflow guards on all exp() calls
- N=0 edge case (system always full)
- λ=0 edge case (empty system)
- Log-space calculations throughout
- M/M/c/N correct for N < c
- Large N capped with detailed warning
- Flow balance sanity checks
- M/M/∞ tail mass reporting

Supported Models:
- M/M/1:GD/∞/∞ - Single server, infinite capacity
- M/M/1:GD/N/∞ - Single server, finite capacity N
- M/M/c:GD/∞/∞ - Multiple servers, infinite capacity
- M/M/c:GD/N/∞ - Multiple servers, finite capacity N
- M/M/∞:GD/∞/∞ - Infinite servers (self-service)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

# Maximum N for finite models to prevent memory/performance issues
MAX_N_ALLOWED = 10000

# IEEE 754 double-precision limits for exp()
EXP_OVERFLOW_THRESHOLD = 709    # exp(710) overflows
EXP_UNDERFLOW_THRESHOLD = -745  # exp(-746) underflows to 0


def safe_exp(log_val: float) -> float:
    """
    Compute exp(log_val) with overflow and underflow protection.
    Returns inf for overflow, 0.0 for underflow.
    """
    if log_val > EXP_OVERFLOW_THRESHOLD:
        return float('inf')
    elif log_val < EXP_UNDERFLOW_THRESHOLD:
        return 0.0
    else:
        return math.exp(log_val)


@dataclass
class QueueingResult:
    """Results from queueing model calculation."""
    model_name: str
    P0: float                      # Probability system is empty
    Pn: Optional[float]            # Probability system is full (for finite models)
    Ls: float                      # Expected number in system
    Lq: float                      # Expected number in queue
    Ws: float                      # Expected time in system
    Wq: float                      # Expected time in queue
    C_bar: float                   # Expected number being served
    lambda_eff: float              # Effective arrival rate
    lambda_lost: float             # Lost customer rate
    probabilities: List[float]     # P(n) for n = 0, 1, 2, ...
    is_stable: bool = True
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    tail_mass: float = 0.0         # Probability mass beyond displayed probabilities


def log_factorial(n: int) -> float:
    """Calculate log(n!) using log-gamma for numerical stability."""
    if n <= 0:
        return 0.0
    return math.lgamma(n + 1)


def factorial(n: int) -> float:
    """Calculate factorial of n (for small n)."""
    if n <= 0:
        return 1.0
    result = 1.0
    for i in range(1, n + 1):
        result *= i
    return result


def log_sum_exp(log_values: List[float]) -> float:
    """
    Compute log(sum(exp(log_values))) in a numerically stable way.
    """
    if not log_values:
        return float('-inf')
    max_val = max(log_values)
    if max_val == float('-inf'):
        return float('-inf')
    if max_val == float('inf'):
        return float('inf')
    return max_val + math.log(sum(math.exp(v - max_val) for v in log_values))


def _make_error_result(model_name: str, error_message: str) -> QueueingResult:
    """Create an error result with NaN values."""
    return QueueingResult(
        model_name=model_name,
        P0=float('nan'), Pn=float('nan'),
        Ls=float('nan'), Lq=float('nan'),
        Ws=float('nan'), Wq=float('nan'),
        C_bar=float('nan'),
        lambda_eff=float('nan'), lambda_lost=float('nan'),
        probabilities=[],
        is_stable=False,
        error_message=error_message
    )


def _validate_int(value, name: str, min_val: int = 0) -> Optional[str]:
    """Validate that value is integer-like and >= min_val."""
    try:
        int_val = int(value)
        if int_val != value:
            return f"Invalid input: {name} must be an integer"
        if int_val < min_val:
            return f"Invalid input: {name} must be ≥ {min_val}"
        return None
    except (TypeError, ValueError):
        return f"Invalid input: {name} must be an integer"


class QueueingModel:
    """
    Queueing theory calculator supporting various Kendall notation models.
    
    Parameters:
        lambda_rate: Arrival rate (λ) - must be >= 0
        mu_rate: Service rate (μ) - must be > 0
    """
    
    def __init__(self, lambda_rate: float, mu_rate: float):
        self.lambda_rate = float(lambda_rate)
        self.mu_rate = float(mu_rate)
    
    def _validate_inputs(self, model_name: str) -> Optional[QueueingResult]:
        """Validate inputs, return error result if invalid, None if valid."""
        if self.lambda_rate < 0:
            return _make_error_result(model_name, "Invalid input: λ must be ≥ 0")
        if self.mu_rate <= 0:
            return _make_error_result(model_name, "Invalid input: μ must be > 0")
        return None
    
    def mm1_infinite(self, max_n: int = 20) -> QueueingResult:
        """
        M/M/1:GD/∞/∞ - Single server, infinite queue, infinite population.
        
        Args:
            max_n: Maximum n for probability distribution (default 20)
        """
        model_name = "M/M/1:GD/∞/∞"
        lam, mu = self.lambda_rate, self.mu_rate
        
        # Validate inputs
        error = self._validate_inputs(model_name)
        if error:
            return error
        
        # Validate max_n
        err_msg = _validate_int(max_n, "max_n", 0)
        if err_msg:
            return _make_error_result(model_name, err_msg)
        max_n = int(max_n)
        
        # Handle λ = 0 edge case (empty system)
        if lam == 0:
            return QueueingResult(
                model_name=model_name,
                P0=1.0, Pn=None, Ls=0.0, Lq=0.0,
                Ws=0.0, Wq=0.0, C_bar=0.0,
                lambda_eff=0.0, lambda_lost=0.0,
                probabilities=[1.0] + [0.0] * max_n
            )
        
        rho = lam / mu
        
        # Check stability condition
        if lam >= mu:
            return QueueingResult(
                model_name=model_name,
                P0=0.0, Pn=None, Ls=float('inf'), Lq=float('inf'),
                Ws=float('inf'), Wq=float('inf'), C_bar=float('inf'),
                lambda_eff=lam, lambda_lost=0.0,
                probabilities=[],
                is_stable=False, error_message="System is unstable (λ ≥ μ)"
            )
        
        # Calculate metrics
        P0 = 1 - rho
        Ls = lam / (mu - lam)
        Lq = (lam ** 2) / (mu * (mu - lam))
        Ws = 1 / (mu - lam)
        Wq = Lq / lam
        C_bar = rho
        
        # Probability distribution in log-space
        log_P0 = math.log(P0)
        log_rho = math.log(rho)
        probabilities = []
        for i in range(max_n + 1):
            log_prob = log_P0 + i * log_rho
            probabilities.append(safe_exp(log_prob))
        
        # Tail mass for infinite queue
        tail_mass = 1.0 - sum(probabilities)
        
        return QueueingResult(
            model_name=model_name,
            P0=P0, Pn=None, Ls=Ls, Lq=Lq, Ws=Ws, Wq=Wq,
            C_bar=C_bar, lambda_eff=lam, lambda_lost=0.0,
            probabilities=probabilities,
            tail_mass=max(0.0, tail_mass)
        )
    
    def mm1_finite(self, N: int, max_n: int = 20) -> QueueingResult:
        """
        M/M/1:GD/N/∞ - Single server, finite capacity N, infinite population.
        
        Args:
            N: System capacity (maximum customers in system)
            max_n: Maximum n for probability distribution
        """
        model_name = "M/M/1:GD/N/∞"
        lam, mu = self.lambda_rate, self.mu_rate
        
        # Validate inputs
        error = self._validate_inputs(model_name)
        if error:
            return error
        
        # Validate N and max_n
        for val, name in [(N, "N"), (max_n, "max_n")]:
            err_msg = _validate_int(val, name, 0)
            if err_msg:
                return _make_error_result(model_name, err_msg)
        N, max_n = int(N), int(max_n)
        original_N = N
        
        # Handle N = 0 edge case (system is always full)
        if N == 0:
            return QueueingResult(
                model_name=model_name,
                P0=1.0, Pn=1.0,
                Ls=0.0, Lq=0.0,
                Ws=0.0, Wq=0.0, C_bar=0.0,
                lambda_eff=0.0, lambda_lost=lam,
                probabilities=[1.0]
            )
        
        # Handle λ = 0 edge case
        if lam == 0:
            return QueueingResult(
                model_name=model_name,
                P0=1.0, Pn=0.0, Ls=0.0, Lq=0.0,
                Ws=0.0, Wq=0.0, C_bar=0.0,
                lambda_eff=0.0, lambda_lost=0.0,
                probabilities=[1.0] + [0.0] * min(N, max_n)
            )
        
        # Cap N for performance
        warning = None
        if N > MAX_N_ALLOWED:
            warning = f"N={original_N} capped to {MAX_N_ALLOWED} for performance; results approximate"
            N = MAX_N_ALLOWED
        
        rho = lam / mu
        
        # Use log-space calculation for stability
        if abs(rho - 1) < 1e-10:
            P0 = 1.0 / (N + 1)
            Pn = P0
            probabilities = [P0] * (min(N, max_n) + 1)
            Ls = N / 2.0
        else:
            log_rho = math.log(rho)
            log_weights = [i * log_rho for i in range(N + 1)]
            log_sum = log_sum_exp(log_weights)
            
            all_probs = [safe_exp(lw - log_sum) for lw in log_weights]
            P0 = all_probs[0]
            Pn = all_probs[N]
            probabilities = all_probs[:min(N + 1, max_n + 1)]
            Ls = sum(n * all_probs[n] for n in range(N + 1))
        
        lambda_eff = lam * (1 - Pn)
        lambda_lost = lam * Pn
        C_bar = lambda_eff / mu if lambda_eff > 0 else 0.0
        Lq = max(0.0, Ls - C_bar)
        Ws = Ls / lambda_eff if lambda_eff > 0 else 0.0
        Wq = Lq / lambda_eff if lambda_eff > 0 else 0.0
        
        # Flow balance sanity check: λ_eff ≈ μ*(1-P0)
        expected_lambda_eff = mu * (1 - P0)
        if lambda_eff > 0 and abs(lambda_eff - expected_lambda_eff) / lambda_eff > 1e-6:
            if warning:
                warning += "; flow balance warning"
            else:
                warning = "Flow balance check failed (numerical issue)"
        
        return QueueingResult(
            model_name=model_name,
            P0=P0, Pn=Pn, Ls=Ls, Lq=Lq, Ws=Ws, Wq=Wq,
            C_bar=C_bar, lambda_eff=lambda_eff, lambda_lost=lambda_lost,
            probabilities=probabilities,
            warning_message=warning
        )
    
    def mmc_infinite(self, c: int, max_n: int = 20) -> QueueingResult:
        """
        M/M/c:GD/∞/∞ - Multiple servers, infinite queue, infinite population.
        
        Args:
            c: Number of servers
            max_n: Maximum n for probability distribution
        """
        model_name = "M/M/c:GD/∞/∞"
        lam, mu = self.lambda_rate, self.mu_rate
        
        # Validate inputs
        error = self._validate_inputs(model_name)
        if error:
            return error
        
        # Validate c and max_n
        err_msg = _validate_int(c, "c", 1)
        if err_msg:
            return _make_error_result(model_name, err_msg)
        err_msg = _validate_int(max_n, "max_n", 0)
        if err_msg:
            return _make_error_result(model_name, err_msg)
        c, max_n = int(c), int(max_n)
        
        # Handle λ = 0 edge case
        if lam == 0:
            return QueueingResult(
                model_name=model_name,
                P0=1.0, Pn=None, Ls=0.0, Lq=0.0,
                Ws=0.0, Wq=0.0, C_bar=0.0,
                lambda_eff=0.0, lambda_lost=0.0,
                probabilities=[1.0] + [0.0] * max_n
            )
        
        a = lam / mu
        rho = lam / (c * mu)
        
        # Check stability condition
        if rho >= 1:
            return QueueingResult(
                model_name=model_name,
                P0=0.0, Pn=None, Ls=float('inf'), Lq=float('inf'),
                Ws=float('inf'), Wq=float('inf'), C_bar=float('inf'),
                lambda_eff=lam, lambda_lost=0.0,
                probabilities=[],
                is_stable=False, error_message="System is unstable (λ/(c×μ) ≥ 1)"
            )
        
        # Calculate log(P0)
        log_a = math.log(a)
        log_terms_before_c = [n * log_a - log_factorial(n) for n in range(c)]
        log_last_term = c * log_a - log_factorial(c) - math.log(1 - rho)
        log_inv_P0 = log_sum_exp(log_terms_before_c + [log_last_term])
        log_P0 = -log_inv_P0
        
        P0 = safe_exp(log_P0)
        
        # Probability distribution
        probabilities = []
        for i in range(max_n + 1):
            if i < c:
                log_prob = log_P0 + i * log_a - log_factorial(i)
            else:
                log_prob = log_P0 + i * log_a - log_factorial(c) - (i - c) * math.log(c)
            probabilities.append(safe_exp(log_prob))
        
        # Calculate Lq with overflow/underflow protection
        log_Lq = log_P0 + c * log_a + math.log(rho) - log_factorial(c) - 2 * math.log(1 - rho)
        Lq = safe_exp(log_Lq)
        
        Ls = Lq + a
        Wq = Lq / lam
        Ws = Ls / lam
        C_bar = a
        
        tail_mass = 1.0 - sum(probabilities)
        
        return QueueingResult(
            model_name=model_name,
            P0=P0, Pn=None, Ls=Ls, Lq=Lq, Ws=Ws, Wq=Wq,
            C_bar=C_bar, lambda_eff=lam, lambda_lost=0.0,
            probabilities=probabilities,
            tail_mass=max(0.0, tail_mass)
        )
    
    def mmc_finite(self, c: int, N: int, max_n: int = 20) -> QueueingResult:
        """
        M/M/c:GD/N/∞ - Multiple servers, finite capacity N, infinite population.
        
        Args:
            c: Number of servers
            N: System capacity (maximum customers in system)
            max_n: Maximum n for probability distribution
        """
        model_name = "M/M/c:GD/N/∞"
        lam, mu = self.lambda_rate, self.mu_rate
        
        # Validate inputs
        error = self._validate_inputs(model_name)
        if error:
            return error
        
        # Validate c, N, max_n
        for val, name, min_v in [(c, "c", 1), (N, "N", 0), (max_n, "max_n", 0)]:
            err_msg = _validate_int(val, name, min_v)
            if err_msg:
                return _make_error_result(model_name, err_msg)
        c, N, max_n = int(c), int(N), int(max_n)
        original_N = N
        
        # Handle N = 0 edge case
        if N == 0:
            return QueueingResult(
                model_name=model_name,
                P0=1.0, Pn=1.0,
                Ls=0.0, Lq=0.0,
                Ws=0.0, Wq=0.0, C_bar=0.0,
                lambda_eff=0.0, lambda_lost=lam,
                probabilities=[1.0]
            )
        
        # Handle λ = 0 edge case
        if lam == 0:
            return QueueingResult(
                model_name=model_name,
                P0=1.0, Pn=0.0, Ls=0.0, Lq=0.0,
                Ws=0.0, Wq=0.0, C_bar=0.0,
                lambda_eff=0.0, lambda_lost=0.0,
                probabilities=[1.0] + [0.0] * min(N, max_n)
            )
        
        # Cap N for performance
        warning = None
        if N > MAX_N_ALLOWED:
            warning = f"N={original_N} capped to {MAX_N_ALLOWED} for performance; results approximate"
            N = MAX_N_ALLOWED
        
        # Birth-death log-weights
        log_lam = math.log(lam)
        log_mu = math.log(mu)
        
        log_weights = [0.0]
        for n in range(1, N + 1):
            servers_active = min(n, c)
            log_service_rate = math.log(servers_active) + log_mu
            log_wn = log_weights[-1] + log_lam - log_service_rate
            log_weights.append(log_wn)
        
        log_sum = log_sum_exp(log_weights)
        all_probs = [safe_exp(lw - log_sum) for lw in log_weights]
        P0 = all_probs[0]
        Pn = all_probs[N]
        probabilities = all_probs[:min(N + 1, max_n + 1)]
        
        lambda_eff = lam * (1 - Pn)
        lambda_lost = lam * Pn
        
        Ls = sum(n * all_probs[n] for n in range(N + 1))
        
        if N <= c:
            Lq = 0.0
        else:
            Lq = sum((n - c) * all_probs[n] for n in range(c + 1, N + 1))
        
        C_bar = Ls - Lq
        Ws = Ls / lambda_eff if lambda_eff > 0 else 0.0
        Wq = Lq / lambda_eff if lambda_eff > 0 else 0.0
        
        # Flow balance sanity check: λ_eff ≈ μ*C_bar
        expected_lambda_eff = mu * C_bar
        if lambda_eff > 0 and abs(lambda_eff - expected_lambda_eff) / lambda_eff > 1e-6:
            if warning:
                warning += "; flow balance warning"
            else:
                warning = "Flow balance check failed (numerical issue)"
        
        return QueueingResult(
            model_name=model_name,
            P0=P0, Pn=Pn, Ls=Ls, Lq=Lq, Ws=Ws, Wq=Wq,
            C_bar=C_bar, lambda_eff=lambda_eff, lambda_lost=lambda_lost,
            probabilities=probabilities,
            warning_message=warning
        )
    
    def mm_infinity(self, max_n: int = 20) -> QueueingResult:
        """
        M/M/∞:GD/∞/∞ - Infinite servers (self-service model).
        
        For large λ/μ, probabilities may appear all zeros; check tail_mass.
        
        Args:
            max_n: Maximum n for probability distribution
        """
        model_name = "M/M/∞:GD/∞/∞"
        lam, mu = self.lambda_rate, self.mu_rate
        
        # Validate inputs
        error = self._validate_inputs(model_name)
        if error:
            return error
        
        err_msg = _validate_int(max_n, "max_n", 0)
        if err_msg:
            return _make_error_result(model_name, err_msg)
        max_n = int(max_n)
        
        # Handle λ = 0 edge case
        if lam == 0:
            return QueueingResult(
                model_name=model_name,
                P0=1.0, Pn=None, Ls=0.0, Lq=0.0,
                Ws=0.0, Wq=0.0, C_bar=0.0,
                lambda_eff=0.0, lambda_lost=0.0,
                probabilities=[1.0] + [0.0] * max_n
            )
        
        a = lam / mu
        log_a = math.log(a)
        
        log_P0 = -a
        P0 = safe_exp(log_P0)
        
        Ls = a
        Lq = 0.0
        Ws = 1.0 / mu
        Wq = 0.0
        C_bar = a
        
        # Poisson probabilities
        probabilities = []
        for n in range(max_n + 1):
            log_prob = -a + n * log_a - log_factorial(n)
            probabilities.append(safe_exp(log_prob))
        
        # Tail mass (important for large a with small max_n)
        tail_mass = 1.0 - sum(probabilities)
        
        # Warning if probabilities look all zeros but shouldn't
        warning = None
        if sum(probabilities) < 0.01 and a > 1:
            warning = f"Most probability mass is around n≈{int(a)}; increase max_n or check tail_mass={tail_mass:.4f}"
        
        return QueueingResult(
            model_name=model_name,
            P0=P0, Pn=None, Ls=Ls, Lq=Lq, Ws=Ws, Wq=Wq,
            C_bar=C_bar, lambda_eff=lam, lambda_lost=0.0,
            probabilities=probabilities,
            tail_mass=max(0.0, tail_mass),
            warning_message=warning
        )
