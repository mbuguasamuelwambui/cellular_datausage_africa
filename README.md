# Nationwide Cellular Telemetry: Diagnostic Analytics & Information-Theoretic Optimization

This repository houses a unified codebase and LaTeX manuscript for analyzing, diagnosing, and optimizing resource-constrained mobile cellular infrastructures. Grounded in carrier-grade physical telemetry from the 2015 Airtel Rwanda trace data (spanning 55TB of traffic signaling logs, latency distributions, and device specifications), this project bridges **forensic empirical diagnostics** with **formal information-theoretic optimization**.

---

## 1. Project Overview & Architectural Flow

The project is structured around two core thematic tracks:

1. **Diagnostic Expansion of Cellular Data Usage:** Analysis of physical infrastructure realities—such as packet loss asymmetry, radio link control (RLC) reordering, TCP bufferbloat, Google edge routing step-functions, spatial node clustering, and diurnal demand fluctuations.
2. **Unified Information-Theoretic Framework:** A three-pillar mathematical formulation modeling the RAN edge, transport transit path, and core gateway signaling processor to locate fundamental transmission bounds.

```
[ Step 1: Raw Telemetry Logs ] ──> [ Step 2: fynesse Diagnostic Engine ] ──> [ Step 3: Math Optimization ] ──> [ Step 4: Academic Output ]
 - qos_g_u_device.xlsx             - fynesse/access.py (Ingestion)            - assess.py (Pillar engines)       - main.tex (LaTeX manuscript)
 - rtt.xlsx (LAN/WAN distributions) - fynesse/assess.py (Plotting/Telemetry)   - main.py (Synthesis & plots)      - Academic plots (output_plots/)
 - rw.gtpc.xlsx (Core signaling)
```

---

## 2. Part 1: Diagnostic Expansion of Cellular Data Usage

Our diagnostics analyze the Airtel Rwanda cellular dataset across the edge, transport, and core layers using the custom `fynesse` library:
* **Asymmetric Throughput \& Latency:** Models throughput imbalances and latency distributions across distinct Radio Access Technologies (2G GSM, 3G UMTS).
* **Out-of-Order Packet Illusion:** Demonstrates that up to 93% of packets on 2G downlinks appear out-of-order due to lower-layer Radio Link Control (RLC) retransmission buffering, misleading transport-layer TCP engines.
* **TCP Bufferbloat \& Satellite Transit Wall:** Isolates the deterministic 200ms round-trip-time (RTT) propagation wall introduced by satellite backhaul routing and quantifies queueing delays.
* **Google Edge Routing \& CDN Latency:** Tracks packet routing hops to Google CDNs and other major internet domains to isolate infrastructural routing overhead.
* **Spatial Node Clustering \& Diurnal Dynamics:** Clusters base stations using spatial and volumetric traffic statistics, mapping diurnal patterns such as the signaling-overhead "ghost hour" tax.

---

## 3. Part 2: Unified Information-Theoretic Framework

The mathematical engines formulate cellular transport as a concatenated information pipeline across three pillars:

### Pillar I: RAN Edge Feedback Exponents with Markovian Delay
We model the wireless feedback delay $d(S_t)$ as a random variable controlled by the Radio Resource Control (RRC) state transition matrix:
$$\mathcal{S} = \{\text{Dedicated (D)}, \text{Shared (S)}, \text{Idle (I)}\}$$
We prove that while Markovian delay does not alter the asymptotic Shannon Capacity, it dictates the lower bounds on the feedback error exponents $E_{\text{fb}}(R, \mathbf{P}_S)$ necessary to prevent premature TCP handshake connection drops.

### Pillar II: Finite-Blocklength (FBL) Concatenated Tandem Bounds
Due to small control-packet blocklengths ($n \in [40, 300]$ bytes), we apply the Polyanskiy-Poor-Verdú (PPV) normal approximation:
$$\log_2 M^*(n, \epsilon) \approx n C_{\text{tandem}} - \sqrt{n V_{\text{tandem}}} Q^{-1}(\epsilon) + \frac{1}{2} \log_2 n$$
We estimate the composite channel dispersion $V_{\text{tandem}}$ and achievable rate limits over the concatenated wireless-satellite transport path under a 200ms latency step-function.

### Pillar III: Semantic Age of Information (AoI) Core Signaling Optimization
We model the core Gateway GPRS Support Node (GGSN) signaling queue as a finite-capacity $M/GI/1/K$ queue. We solve for the optimal update rate $\lambda^*$ that minimizes the Peak Age of Information (PAoI):
$$\mathbb{E}[\Delta_{\text{peak}}] = \mathbb{E}[T] + \frac{1}{\lambda_{\text{sig}}(1 - P_{\text{drop}})}$$
This prevents signaling saturation from cheap, low-payload "chatty" mobile devices.

---

## 4. Repository Structure

```
dsail/research/cellular/rwanda/
├── fynesse/                      # Core analysis library
│   ├── __init__.py               # Package initialization
│   ├── access.py                 # ETL Ingestion & parameter extraction
│   ├── assess.py                 # Telemetry analysis & numerical engines
│   ├── address.py                # Spatial node clustering
│   └── infotheory.py             # Basic information theory routines
├── notebook/                     # Jupyter notebooks for interactive execution
│   ├── main.ipynb                # Restructured experiment runner
│   ├── info_theory.ipynb         # Information theory diagnostic notebook
│   └── rwanda_cellular2015 (1).ipynb # Baseline notebook
├── output_plots/                 # Directory containing generated high-res plots (300 dpi)
│   ├── fig1_error_exponent_markov.png   # Exponent comparison vs. transmission rate
│   ├── fig2_fbl_satellite_tandem.png     # Achievable FBL rate vs. blocklength
│   └── fig3_aoi_signaling_saturation.png # Peak AoI vs. core signaling load
├── main.py                       # Master execution script (Phase 3 Synthesis)
├── main.tex                      # IEEE Transactions double-column LaTeX manuscript
├── requirements.txt              # Clean python dependency requirements
└── .gitignore                    # Local environment and data ignores
```

---

## 5. Execution & Replication

### Prerequisites & Dependency Setup
Verify that python is installed, then set up the environment and install requirements:
```bash
# Clone the repository
git clone https://github.com/mbuguasamuelwambui/cellular_datausage_africa.git
cd cellular_datausage_africa

# Create a virtual environment
python -m venv cellular_env
source cellular_env/bin/activate  # On Windows use: cellular_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Joint Optimization Engine
To extract empirical parameters from the cellular trace files, run the mathematical model simulation, and regenerate the academic figures:
```bash
python main.py
```
This will print out the calculated RRC state probabilities, satellite channel dispersion, and optimal signaling update rates, and write the three figures (`fig1_error_exponent_markov.png`, `fig2_fbl_satellite_tandem.png`, `fig3_aoi_signaling_saturation.png`) to `output_plots/`.

### LaTeX Manuscript Compilation
Compile the LaTeX document `main.tex` to generate the PDF manuscript draft conforming to the IEEE double-column layout:
```bash
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
```
