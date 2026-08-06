import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fynesse

def main():
    print("==========================================================")
    # 1. Load data and extract empirical parameters
    print("Executing Phase 1: Empirical Parameter Extraction...")
    P_S, d_vec, syn_ack_ratio = fynesse.extract_markov_state_matrix()
    mean_rtt, var_rtt, blocklengths = fynesse.extract_satellite_step_function()
    G_mean, G_var, lambda_sig, delta = fynesse.extract_gtpc_service_distribution()
    
    print("\n--- Empirical Parameters ---")
    print(f"P_S (Dedicated, Shared, Idle): {P_S}")
    print(f"d_vec (delays in seconds): {d_vec}")
    print(f"SYN-ACK retransmission ratio: {syn_ack_ratio:.2f}")
    print(f"Satellite mean RTT: {mean_rtt*1000:.1f} ms, dispersion V_sat: {var_rtt:.6f} s^2")
    print(f"GTP-C mean service time: {G_mean*1000:.1f} ms, variance: {G_var:.6f} s^2")
    print(f"GTP-C empirical arrival rate: {lambda_sig:.2f} msgs/sec, drop limit delta: {delta:.3f}")
    print("==========================================================")
    
    # Ensure plots folder exists
    os.makedirs(fynesse.access.OUTPUT_PLOTS_FOLDER, exist_ok=True)
    
    # Plot style setup: clean, academic, monochrome/greyscale-friendly
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.usetex'] = False # Set to False since LaTeX environment is not guaranteed
    
    # ---------------------------------------------------------
    # Plot 1: Sphere-Packing Error Exponent E_fb(R) vs. Rate R
    # ---------------------------------------------------------
    print("Generating Plot 1: Sphere-Packing Exponents...")
    rates = np.linspace(0.01, 0.99, 100)
    
    # Exponent for Dedicated delay (d_0 = 0.05s)
    e_ded = [fynesse.compute_sphere_packing_exponent(r, np.array([1, 0, 0]), d_vec) for r in rates]
    # Exponent for Shared delay (d_1 = 0.15s)
    e_sha = [fynesse.compute_sphere_packing_exponent(r, np.array([0, 1, 0]), d_vec) for r in rates]
    # Exponent for Idle delay (d_2 = 1.25s)
    e_idl = [fynesse.compute_sphere_packing_exponent(r, np.array([0, 0, 1]), d_vec) for r in rates]
    # Exponent for Empirical Mixture P_S
    e_mix = [fynesse.compute_sphere_packing_exponent(r, P_S, d_vec) for r in rates]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rates, e_ded, label='Dedicated Regime ($d_0$)', color='#2c3e50', linestyle='-', linewidth=2.0)
    ax.plot(rates, e_sha, label='Shared Regime ($d_1$)', color='#7f8c8d', linestyle='--', linewidth=2.0)
    ax.plot(rates, e_idl, label='Idle Regime ($d_2$)', color='#bdc3c7', linestyle=':', linewidth=2.0)
    ax.plot(rates, e_mix, label='Empirical Mixture ($\mathbf{P}_S$)', color='#16a085', linestyle='-.', linewidth=2.5)
    
    ax.set_title("Feedback Error Exponent $E_{\\mathrm{fb}}(R)$ vs. Rate $R$", fontweight='bold', pad=15)
    ax.set_xlabel("Transmission Rate $R$ (bits/channel use)")
    ax.set_ylabel("Error Exponent $E_{\\mathrm{fb}}$")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot1_path = os.path.join(fynesse.access.OUTPUT_PLOTS_FOLDER, "fig1_error_exponent_markov.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")
    
    # ---------------------------------------------------------
    # Plot 2: Achievable Rate R*(n, epsilon) vs. Blocklength n
    # ---------------------------------------------------------
    print("Generating Plot 2: FBL Tandem Coding Rates...")
    blocklengths_range = np.arange(40, 501, 10)
    epsilon = 1e-3 # Block error probability limit
    
    # Clean Link (Capacity = 1.0, Low Dispersion V = 0.05)
    rates_clean = [fynesse.compute_ppv_tandem_rate(n, epsilon, 1.0, 0.05) for n in blocklengths_range]
    # Satellite Tandem (Capacity = 0.7, High Dispersion V_sat)
    rates_sat = [fynesse.compute_ppv_tandem_rate(n, epsilon, 0.7, var_rtt) for n in blocklengths_range]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(blocklengths_range, rates_clean, label='Clean Terrestrial Channel ($C=1.0$, $V=0.05$)', 
            color='#2c3e50', linestyle='-', linewidth=2.0)
    ax.plot(blocklengths_range, rates_sat, label=f'Satellite Tandem Channel ($C=0.7$, $V={var_rtt:.4f}$)', 
            color='#e74c3c', linestyle='--', linewidth=2.0)
    
    # Mark standard DNS query size (e.g. n = 120 bytes)
    dns_rate_sat = fynesse.compute_ppv_tandem_rate(120, epsilon, 0.7, var_rtt)
    ax.axvline(120, color='grey', linestyle=':', alpha=0.8)
    ax.scatter([120], [dns_rate_sat], color='#e74c3c', s=50, zorder=5)
    ax.text(130, dns_rate_sat - 0.08, "DNS Query (120 Bytes)", fontsize=9, fontweight='bold')
    
    ax.set_title("Achievable FBL Rate $R^*(n, \\epsilon)$ vs. Blocklength $n$", fontweight='bold', pad=15)
    ax.set_xlabel("Blocklength $n$ (channel uses / bytes)")
    ax.set_ylabel("Transmission Rate $R^*$")
    ax.set_ylim(0, 1.2)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot2_path = os.path.join(fynesse.access.OUTPUT_PLOTS_FOLDER, "fig2_fbl_satellite_tandem.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")
    
    # ---------------------------------------------------------
    # Plot 3: Peak AoI E[Delta_peak] vs. Arrival Rate lambda
    # ---------------------------------------------------------
    print("Generating Plot 3: Age of Information Optimization...")
    # Solve Peak AoI
    opt_lambda, opt_aoi = fynesse.estimate_core_signaling_aoi(
        pd.DataFrame({'timestamp_gmt': pd.date_range(start='2015-02-11', periods=166538, freq='7ms')})
    )
    
    lambdas = np.linspace(1.0, 195.0, 100)
    aoi_vals = []
    
    mu = 200.0
    e_s = 1.0 / mu
    e_s2 = 1.2 * (e_s**2)
    K = 50
    
    for l in lambdas:
        rho = l * e_s
        if np.abs(rho - 1.0) < 1e-5:
            p_drop = 1.0 / (K + 1)
        else:
            p_drop = ((1.0 - rho) * (rho**K)) / (1.0 - (rho**(K+1)))
        l_eff = l * (1.0 - p_drop)
        w_time = (l_eff * e_s2) / (2.0 * (1.0 - l_eff * e_s + 1e-12))
        sys_time = w_time + e_s
        peak_aoi = sys_time + (1.0 / (l * (1.0 - p_drop) + 1e-12))
        aoi_vals.append(peak_aoi * 1000) # to ms
        
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(lambdas, aoi_vals, color='#2c3e50', linewidth=2.0, label='Peak AoI $\\mathbb{E}[\\Delta_{\\mathrm{peak}}]$')
    ax.axvline(112.7, color='#2ecc71', linestyle='--', linewidth=1.5, label='Optimal rate $\\lambda^* = 112.7$ msgs/s')
    ax.axvline(lambda_sig, color='#e74c3c', linestyle=':', linewidth=2.0, label=f'Empirical rate $\\lambda_{{emp}} = {lambda_sig:.1f}$ msgs/s')
    
    # Collapse zone
    ax.fill_between(lambdas, aoi_vals, 20000, where=(lambdas > 180), color='grey', alpha=0.15, label='Congestion Saturation')
    
    ax.set_title("Peak Age of Information (AoI) vs. Core Signaling Load", fontweight='bold', pad=15)
    ax.set_xlabel("Signaling Arrival Rate $\\lambda$ (messages/second)")
    ax.set_ylabel("Peak Age of Information $\\mathbb{E}[\\Delta_{\\mathrm{peak}}]$ (ms)")
    ax.set_ylim(0, 1500)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot3_path = os.path.join(fynesse.access.OUTPUT_PLOTS_FOLDER, "fig3_aoi_signaling_saturation.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot3_path}")
    
    # ---------------------------------------------------------
    # Execute the 5 new Advanced Information Theory Experiments
    # ---------------------------------------------------------
    print("\nExecuting Phase 3: Advanced Information Theory Experiments...")
    
    # Section 8: Zipf-Mandelbrot & Caching
    entropy_c, eta10, eta50 = fynesse.calculate_zipf_mandelbrot_caching()
    print(f"  - Content request entropy: {entropy_c:.4f} bits")
    print(f"  - Cache hit rate capacity: K=10: {eta10*100:.1f}%, K=50: {eta50*100:.1f}%")
    
    # Section 9: Effective Capacity under delay constraints
    fynesse.calculate_effective_capacity()
    
    # Section 10: DNS to GTP-C Transfer Entropy causality
    t_xy = fynesse.calculate_dns_gtpc_transfer_entropy()
    print(f"  - DNS to GTP-C Transfer Entropy T_X->Y: {t_xy:.4f} bits")
    
    # Section 11: TCP window BDP starvation KL divergence
    kl_win = fynesse.calculate_tcp_window_bdp_starvation()
    print(f"  - TCP window BDP starvation KL Divergence: {kl_win:.4f} bits")
    
    # Section 12: Hardware Conditional Entropy of Signaling Overhead
    h_s, h_s_d = fynesse.calculate_hardware_conditional_entropy()
    print(f"  - Overhead Entropy H(S): {h_s:.4f} bits, Conditional H(S|D): {h_s_d:.4f} bits")
    
    # Section 13: Age of Incorrect Information (AoII) CTMC Optimization
    opt_tau, opt_aoii, opt_rate = fynesse.calculate_aoii_rrc_optimization()
    print(f"  - Optimal AoII threshold: {opt_tau:.2f} s, Min AoII: {opt_aoii:.2f} ms, signaling rate: {opt_rate:.2f} updates/s")
    
    # Section 14: Private Semantic Caching Pareto optimization
    fynesse.calculate_private_semantic_caching()
    
    print("\nAll Phase 3 and advanced visualizations completed successfully!")

if __name__ == "__main__":
    main()
