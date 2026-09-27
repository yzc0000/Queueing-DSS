# Queueing Theory Decision Support System

An interactive Streamlit application for analysing queueing systems using Kendall notation. The application calculates steady-state probabilities and operating measures for single-server, multi-server, finite-capacity, and infinite-server models.

## Supported models

The calculator provides:

- `M/M/1:GD/infinity/infinity` — one server with unlimited waiting capacity
- `M/M/1:GD/N/infinity` — one server with finite system capacity `N`
- `M/M/c:GD/infinity/infinity` — `c` parallel servers with unlimited capacity
- `M/M/c:GD/N/infinity` — `c` parallel servers with finite system capacity `N`
- `M/M/infinity:GD/infinity/infinity` — infinite-server or self-service model

For each model, the user supplies an arrival rate (`lambda`), service rate (`mu`), and model-specific parameters such as server count, capacity, and the displayed probability range.

## Reported measures

The model returns:

- `P0`: probability that the system is empty
- `Pn`: full-system probability for finite-capacity models
- `Ls`: expected number of customers in the system
- `Lq`: expected number waiting in the queue
- `Ws`: expected time in the system
- `Wq`: expected waiting time
- `C_bar`: expected number of busy servers
- `lambda_eff`: effective arrival rate
- `lambda_lost`: rejected or lost arrival rate
- `P(n)`: displayed state probabilities and, for infinite models, the unshown tail mass

The Streamlit interface also provides a probability-distribution chart, a downloadable probability table, model interpretation, and warnings for unstable systems or high blocking.

## Numerical safeguards

The calculation module is designed for edge cases and large parameters:

- Input validation rejects negative arrival rates, non-positive service rates, invalid server counts, and invalid capacities.
- Exponentials are evaluated through overflow and underflow guards.
- Probability calculations use log-space arithmetic where large factorials or state spaces could overflow.
- Finite capacities larger than 10,000 are capped with an explicit warning.
- `N=0` and `lambda=0` are handled explicitly.
- Flow-balance checks compare effective arrivals with service throughput.
- Infinite-capacity models report tail mass when the displayed probability range does not contain all probability.

## Run the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application opens at `http://localhost:8501`.

## Verification

Run the numerical verification suite with UTF-8 output enabled on Windows:

```bash
python verify_models.py
```

The current run passed all **8/8 verification groups**:

| Verification | Result |
|---|---|
| Overflow and underflow guards | Passed: `exp(-800) -> 0`, `exp(800) -> inf` |
| Input validation | Passed for invalid rates, server count, capacity, and display range |
| `N=0` edge cases | Passed for `M/M/1/N` and `M/M/c/N` |
| `M/M/c` near-utilization stability | Passed with `Lq = 999,995.34` at `rho = 0.999999` without crashing |
| Large-capacity handling | Passed; `N=50,000` is capped to `10,000` with a warning |
| Tail-mass reporting | Passed for `M/M/1` and `M/M/infinity` |
| Finite-system flow balance | Passed with relative error `1.12e-16` |
| Standard analytical cases | Passed: `M/M/1` and `M/M/infinity` reference values |

Reference values checked by the suite include:

- `M/M/1`, `lambda=2`, `mu=3`: `P0=0.3333`, `Ls=2.0000`, `Lq=1.3333`, `C_bar=0.6667`
- `M/M/infinity`, `lambda=3`, `mu=2`: `Ls=1.500000`, `Lq=0`, `P0=0.223130`
- `M/M/1`, `rho=0.9`, displayed through `n=5`: tail mass `0.5314`

## Project structure

```text
app.py                 Streamlit user interface and result presentation
queueing_models.py     Queueing formulas, validation, safeguards, and result objects
verify_models.py       Numerical and edge-case verification suite
requirements.txt       Streamlit and Plotly dependencies
```

## Scope

This project is a decision-support calculator for analytical queueing models. Its results assume the arrival and service distributions represented by the selected Kendall notation and should be interpreted accordingly when applied to operational systems.
