# An End-to-End Information-Theoretic Framework for Resource-Constrained Cellular Infrastructure
## Foundational Rationale, Empirical Grounding, and Comprehensive Academic Reading Guide

---

## I. Why We Are Doing This Work (The Core Motivation & Problem Statement)

### 1. The Breakdown of Classical Shannon Information Theory in Emerging Markets
Classical Shannon Information Theory has served as the bedrock of modern telecommunications for over seven decades. However, its fundamental capacity theorems—including $C = \max I(X; Y)$—rely on three idealized asymptotic assumptions that fail when applied to real-world cellular infrastructure in resource-constrained environments:

1. **Instantaneous, Memoryless Feedback:** Classical theory assumes that channel state information and receiver feedback return instantaneously without delay or structural bias.
2. **Asymptotic Infinite Blocklengths ($n \to \infty$):** Shannon capacity theorems assume arbitrarily long codewords where packet header overhead, control-plane handshakes, and propagation latency are mathematically negligible.
3. **Payload-Centric Throughput:** Classical rate optimization prioritizes maximizing user-plane bits per second while treating control-plane signaling overhead as an infinitesimal fraction of total channel capacity.

### 2. The Physical Reality of National-Scale Cellular Networks
Empirical evidence from national-scale telecommunications—specifically the **55 TB Airtel Rwanda 2015 cellular trace** spanning 200,000 subscribers, 152 million DNS queries, and 330,000 core gateway control messages—demonstrates that these classical assumptions are systematically violated in the Global South:

* **Markovian Radio Feedback Latency:** Radio Access Network (RAN) link latency is not memoryless; it oscillates between **400ms and 1500ms** depending on discrete Radio Resource Control (RRC) state transitions (Idle $\leftrightarrow$ Shared $\leftrightarrow$ Dedicated). Consequently, TCP SYN-ACK handshakes account for **90%** of all uplink retransmissions, while local 2G Radio Link Control (RLC) retransmissions create a **93%** out-of-order packet delivery illusion that forces standard TCP congestion control into persistent deflation.
* **The Finite-Blocklength Satellite Backhaul Wall:** Long-haul Wide Area Network (WAN) routing exhibits a deterministic **200ms** Round Trip Time (RTT) step-function across **49.8%** of all flows due to geostationary (GEO) satellite backhaul. Because mobile devices primarily exchange short control packets and brief DNS queries (**60 to 250 bytes**), asymptotic Shannon capacity severely overestimates the true achievable transmission rate.
* **Control-Plane Signaling Saturation (The "Invisible Tax"):** While video streaming dominates total downlink payload volume (**12.97 TB**), low-cost feature phones (OEM brands like TECNO and ITEL, representing **78.8%** of the device census) and "chatty" messaging applications flood the Core Gateway (GGSN/SGSN) with background keep-alive signaling. This generates an empirical **1.2%** unmatched request failure rate where control-plane queue exhaustion collapses session establishment before payload transmission can even begin.

### 3. Why This Elevates Your Work to Top-Tier IEEE Journals
In premier theoretical and networking venues—such as **IEEE Transactions on Information Theory (IEEE TIT)** and **IEEE Journal on Selected Areas in Communications (IEEE JSAC)**—reviewers reject purely descriptive empirical measurement studies. This research bridges the gap between **forensic network measurement** and **mathematical information theory** by establishing:

* **New Converse and Achievability Bounds:** Proving the maximal achievable short-packet rate $R^*(n, \epsilon)$ over concatenated fading-satellite links.
* **Sphere-Packing Error Exponents:** Characterizing reliability $E_{\text{fb}}(R, \mathbf{P}_S)$ under delayed Markovian feedback.
* **Semantic Freshness Limits:** Proving the existence of an optimal arrival rate $\lambda^*$ that minimizes the Peak Age of Information ($\Delta_{\text{peak}}$) and Age of Incorrect Information (AoII) in multi-source $M/GI/1/K$ control-plane queues.

---

## II. End-to-End Codebase & Dataset Grounding Table

| Experiment # | Theoretical Diagnostic Topic | Primary Math / Information Metric | Airtel Rwanda Dataset Source File | Implementation Method in `assess.py` / `fynesse` |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Diurnal Demand Predictability | Shannon Entropy $H(X) = -\sum p_h \log_2 p_h$ | `tcp-udp-bw.xlsx` (`ip_flow_bwhist_g_u_agg`) | `calculate_hourly_traffic_entropy(df)` |
| **2** | Weekday vs. Weekend Shift | Kullback-Leibler Divergence $D_{\text{KL}}(P \parallel Q)$ | `tcp-udp-bw.xlsx` (`ip_flow_bwhist_g_u_agg`) | `calculate_weekday_weekend_kl_divergence(df)` |
| **3** | RAT vs. Traffic Code Coding | Mutual Information $I(\text{RAT}; \text{Code})$ | `tcp-udp-bw.xlsx` (`ip_flow_bwhist_g_u_agg`) | `calculate_rat_traffic_code_mutual_information(df)` |
| **4** | RRC Feedback State Vector | Markov Transition Vector $\mathbf{P}_S = [\pi_D, \pi_S, \pi_I]$ | `qos_g_u_device.xlsx` (`tcp_flow_qos_g_u_device`) | `estimate_feedback_channel_states(qos_df)` |
| **5** | Satellite Tandem Dispersion | Polyanskiy Channel Dispersion $V_{\text{tandem}}$ | `rtt.xlsx` (`wan`), `dnsflowcontent_3dom.txt` | `estimate_satellite_dispersion(wan_df)` |
| **6** | Core GTP-C Queue Freshness | Peak Age of Information $\mathbb{E}[\Delta_{\text{peak}}]$ | `rw.gtpc.xlsx` (`request`, `response`) | `estimate_core_signaling_aoi(gtpc_df)` |
| **7** | Content Popularity Caching | Zipf-Mandelbrot Entropy $H(\text{Content})$ | `content.xlsx` (`down_notld`) | `calculate_zipf_mandelbrot_caching(dns_df, content_df)` |
| **8** | Delay-Constrained Capacity | Effective Capacity $E_c(\theta)$ & QoS Exponents | `rtt.xlsx` (`lan_umts`), `tcp-udp-bw.xlsx` | `calculate_effective_capacity(rtt_df, bw_df)` |
| **9** | DNS-to-Core Signaling Causal | Transfer Entropy (Directed Info) $T_{X \to Y}$ | `dns.xlsx` (`dnshist.txt`), `rw.gtpc.xlsx` | `calculate_dns_gtpc_transfer_entropy(dns_df, gtpc_df)` |
| **10** | TCP Window BDP Starvation | Window Divergence $D_{\text{KL}}(P_{\text{win}} \parallel P_{\text{BDP}})$ | `tcp.xlsx` (`tcpflow_stats_bw.txt`) | `calculate_tcp_window_bdp_starvation(tcp_df)` |
| **11** | Hardware Protocol Overhead | Conditional Entropy $H(S \mid D)$ by TAC Census | `data.xlsx` (`types`), `tac.xlsx` (`tac_top`) | `calculate_hardware_conditional_entropy(qos_df, tac_df)` |
| **12** | Semantic Accuracy over RAN | Age of Incorrect Information (AoII) over CTMC | `qos_g_u_device.xlsx`, `rw.gtpc.xlsx` | `calculate_aoii_rrc_optimization(Q, d_vec)` |
| **13** | Goal-Oriented Edge Caching | Private Semantic Caching Pareto Frontier | `content.xlsx` (`down_notld`), `dest_ip.xlsx` | `calculate_private_semantic_caching(content_df, eps_range)` |
| **14** | Digital Redlining & Lag | Legacy Traps & Infrastructural Lag ($I_{\text{lag}}$) | `tac.xlsx`, `rtt.xlsx`, `qos_g_u_device.xlsx` | `plot_infrastructural_lag()`, `analyze_legacy_traps(df)` |

---

## III. Curated Academic Reading Guide & Reference Curriculum

### Category 1: Finite-Blocklength (FBL) Information Theory & Dispersion
1. **Polyanskiy, Y., Poor, H. V., & Verdú, S. (2010).** "Channel Coding Rate in the Finite Blocklength Regime." *IEEE Transactions on Information Theory*, 56(5), 2307–2359.
   * **Key Concept:** Defines Channel Dispersion (V) and the normal approximation for short-packet transmissions.
   * **URL:** https://www.mit.edu/~medard/Itpaperaward/review_Polyanskiy_Poor_Verdu.pdf
2. **Yang, W., Durisi, G., Koch, T., & Polyanskiy, Y. (2016).** "Finite-Blocklength Information Theory: What is the Practical Impact on Wireless Communications?" *IEEE Globecom Workshops / IEEE Trans. on Communications*.

### Category 2: Semantic Information Theory — Age of Information (AoI) & AoII
3. **Kaul, S. K., Yates, R. D., & Gruteser, M. (2012).** "Real-Time Status: How Often Should One Update?" *Proceedings of IEEE INFOCOM*, pp. 2731–2735.
   * **Key Concept:** Invented Age of Information ($\Delta(t) = t - u(t)$) and proved that updating too frequently causes queueing congestion that increases staleness.
   * **URL:** https://www.winlab.rutgers.edu/~gruteser/papers/sanjitnew.pdf
4. **Maatouk, A., Saad, S., Assaad, M., & Ephremides, A. (2020).** "The Age of Incorrect Information: A New Performance Metric for Status Updates." *IEEE Transactions on Networking*.
   * **URL:** https://www.emergentmind.com/topics/age-of-incorrect-information-aoii

### Category 3: Effective Capacity & Quality of Service (QoS) Exponents
5. **Wu, D., & Negi, R. (2003).** "Effective Capacity: A Wireless Link Model for Support of Quality of Service." *IEEE Transactions on Wireless Communications*, 2(4), 630–643.
   * **Key Concept:** Bridges physical-layer fading and network-layer buffer overflow using large deviations theory and QoS exponent $\theta$.
   * **URL:** https://users.ece.cmu.edu/~negi/publications/preprints/JSAC_camera.pdf

### Category 4: Empirical Grounding — The 2015 Airtel Rwanda Trace
6. **Akoush, S., Sathiaseelan, A., & Crowcroft, J. (2015).** "Cellular Traffic in Developing Nations." *ACM DEV / Cambridge Computer Laboratory Technical Report*.
   * **Key Concept:** Baseline dataset paper documenting the 55 TB Rwanda trace, 1:10 upload/download asymmetry, and OEM device distributions.
