import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import access

def calculate_hourly_traffic_entropy(df):
    """
    Computes Shannon Entropy of the diurnal traffic load distribution.
    A higher entropy indicates uniform distribution of network demand throughout the day,
    while a lower entropy indicates traffic concentrated in specific peak hours.
    """
    print("Computing Shannon Entropy of diurnal traffic load...")
    
    # Sum values by hour of day
    hourly_vols = df.groupby('hour')['val'].sum()
    total_vol = hourly_vols.sum()
    
    if total_vol == 0:
        print("Total traffic volume is zero. Cannot compute entropy.")
        return 0.0
        
    probs = hourly_vols / total_vol
    # Shannon Entropy formula: H(X) = -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    # Theoretical maximum entropy for 24 hours (uniform distribution)
    max_entropy = np.log2(24)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(probs.index, probs.values * 100, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
    
    ax.set_title("Diurnal Traffic Volume Distribution (Probability Share)", fontweight='bold', pad=15)
    ax.set_xlabel("Hour of Day (24h)")
    ax.set_ylabel("Traffic Volume Share (%)")
    ax.set_xticks(range(24))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Text annotation for calculated entropy
    ax.text(0.05, 0.95, f"Shannon Entropy: {entropy:.4f} bits\nTheoretical Max: {max_entropy:.4f} bits\nUniformity Index: {(entropy/max_entropy)*100:.2f}%", 
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='grey'))
            
    plt.tight_layout()
    plot_path = os.path.join(access.OUTPUT_PLOTS_FOLDER, "hourly_traffic_entropy.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated hourly_traffic_entropy.png (Entropy = {entropy:.4f} bits)")
    
    return entropy

def calculate_weekday_weekend_kl_divergence(df):
    """
    Computes the Kullback-Leibler (KL) Divergence between the weekday diurnal traffic distribution (P)
    and weekend diurnal traffic distribution (Q).
    """
    print("Computing Kullback-Leibler (KL) Divergence between weekdays and weekends...")
    
    # Separate traffic
    weekday_df = df[~df['is_weekend']]
    weekend_df = df[df['is_weekend']]
    
    weekday_hourly = weekday_df.groupby('hour')['val'].sum()
    weekend_hourly = weekend_df.groupby('hour')['val'].sum()
    
    if weekday_hourly.sum() == 0 or weekend_hourly.sum() == 0:
        print("Empty dataset for weekdays or weekends. Cannot compute KL divergence.")
        return 0.0
        
    # Get probability distributions
    p = weekday_hourly / weekday_hourly.sum()
    q = weekend_hourly / weekend_hourly.sum()
    
    # Ensure index alignment
    p = p.reindex(range(24), fill_value=0.0)
    q = q.reindex(range(24), fill_value=0.0)
    
    # KL Divergence formula: D_KL(P || Q) = sum(P * log2(P / Q))
    kl_p_q = np.sum(p * np.log2((p + 1e-12) / (q + 1e-12)))
    kl_q_p = np.sum(q * np.log2((q + 1e-12) / (p + 1e-12)))
    
    # Plotting
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(p.index, p.values * 100, label='Weekdays (P)', color='#3498db', linewidth=2.5, marker='o')
    ax.plot(q.index, q.values * 100, label='Weekends (Q)', color='#e74c3c', linewidth=2.5, marker='s', linestyle='--')
    
    ax.set_title("Diurnal Traffic Volume Profile: Weekdays vs. Weekends", fontweight='bold', pad=15)
    ax.set_xlabel("Hour of Day (24h)")
    ax.set_ylabel("Hourly Traffic Share (%)")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Text annotation for KL Divergence
    ax.text(0.05, 0.95, f"D_KL(Weekdays || Weekends): {kl_p_q:.6f} bits\nD_KL(Weekends || Weekdays): {kl_q_p:.6f} bits", 
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='grey'))
            
    plt.tight_layout()
    plot_path = os.path.join(access.OUTPUT_PLOTS_FOLDER, "weekday_weekend_kl_divergence.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated weekday_weekend_kl_divergence.png (D_KL = {kl_p_q:.6f} bits)")
    
    return kl_p_q

def calculate_rat_traffic_code_mutual_information(df):
    """
    Computes the Mutual Information between Radio Access Technology (RAT) and traffic type/direction (code).
    """
    print("Computing Mutual Information between RAT and Traffic Code...")
    
    # Filter code to known payload/signaling codes
    code_map = {
        'dt': 'Downlink Payload',
        'ut': 'Uplink Payload',
        'dm': 'Downlink Signaling',
        'um': 'Uplink Signaling'
    }
    filtered_df = df[df['code'].isin(code_map.keys())].copy()
    filtered_df['traffic_type'] = filtered_df['code'].map(code_map)
    
    rat_map = {1: '2G (GPRS/EDGE)', 2: '2.5G (EDGE)', 3: '3G (UMTS)', 4: '3.5G (HSPA)'}
    filtered_df['network_gen'] = filtered_df['rat'].map(rat_map).fillna(filtered_df['rat'].apply(lambda x: f"RAT {x}"))
    
    # Calculate joint contingency table of traffic volume (val)
    contingency = filtered_df.groupby(['network_gen', 'traffic_type'])['val'].sum().unstack(fill_value=0.0)
    total_val = contingency.values.sum()
    
    if total_val == 0:
        print("Total traffic volume is zero. Cannot compute Mutual Information.")
        return 0.0
        
    # Joint probability distribution p(x, y)
    p_xy = contingency / total_val
    
    # Marginal probability distributions p(x) and p(y)
    p_x = p_xy.sum(axis=1) # RAT marginals
    p_y = p_xy.sum(axis=0) # Code marginals
    
    # Mutual Information formula: I(X; Y) = sum_x sum_y p(x,y) * log2(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for x in contingency.index:
        for y in contingency.columns:
            val_xy = p_xy.loc[x, y]
            val_x = p_x.loc[x]
            val_y = p_y.loc[y]
            if val_xy > 0:
                mi += val_xy * np.log2(val_xy / (val_x * val_y + 1e-12))
                
    # Max theoretical Mutual Information is min(H(X), H(Y))
    h_x = -np.sum(p_x * np.log2(p_x + 1e-12))
    h_y = -np.sum(p_y * np.log2(p_y + 1e-12))
    max_mi = min(h_x, h_y)
    
    # Plotting: Joint Probability Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(p_xy * 100, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=ax,
                cbar_kws={'label': 'Joint Probability Share (%)'})
    
    ax.set_title("Joint Distribution Share: RAT vs. Traffic Type", fontweight='bold', pad=15)
    ax.set_xlabel("Traffic Type / Code")
    ax.set_ylabel("Radio Access Technology (RAT)")
    
    # Add text annotation for Mutual Information
    plt.figtext(0.15, 0.02, f"Mutual Information I(RAT; Code): {mi:.6f} bits  |  Max Potential MI: {max_mi:.6f} bits",
                ha="left", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="grey"))
                
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plot_path = os.path.join(access.OUTPUT_PLOTS_FOLDER, "rat_code_mutual_information.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated rat_code_mutual_information.png (MI = {mi:.6f} bits)")
    
    return mi

def estimate_feedback_channel_states(qos_df):
    """
    Estimates the empirical stationary distribution P_S of RRC states
    based on packet retransmissions and out-of-order sequence flags.
    """
    print("Estimating feedback channel RRC states from QoS telemetry...")
    
    # Define states
    # Dedicated: retransmissions = 0 and out_of_order = 0
    # Shared: retransmissions = 0 and out_of_order > 0
    # Idle: retransmissions > 0
    
    ded = qos_df[(qos_df['retransmission'] == 0) & (qos_df['out_of_order'] == 0)]
    sha = qos_df[(qos_df['retransmission'] == 0) & (qos_df['out_of_order'] > 0)]
    idl = qos_df[qos_df['retransmission'] > 0]
    
    total = len(qos_df)
    p_ded = len(ded) / total
    p_sha = len(sha) / total
    p_idl = len(idl) / total
    
    print(f"Empirical State Distribution P_S:")
    print(f"  - Dedicated (Low Latency): {p_ded*100:.2f}%")
    print(f"  - Shared (RLC Reordering): {p_sha*100:.2f}%")
    print(f"  - Idle (Retransmissions):  {p_idl*100:.2f}%")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 5))
    states = ['Dedicated', 'Shared', 'Idle']
    shares = [p_ded * 100, p_sha * 100, p_idl * 100]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    ax.bar(states, shares, color=colors, edgecolor='black', width=0.4)
    ax.set_title("Empirical RRC Feedback State Distribution P_S", fontweight='bold', pad=15)
    ax.set_ylabel("State Probability Share (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, v in enumerate(shares):
        ax.text(i, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plot_path = os.path.join(access.OUTPUT_PLOTS_FOLDER, "rrc_feedback_states.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print("Generated rrc_feedback_states.png")
    
    return [p_ded, p_sha, p_idl]

def estimate_satellite_dispersion(wan_df):
    """
    Quantifies the WAN satellite propagation delay wall (200ms)
    and computes the empirical channel dispersion V_sat.
    """
    print("Estimating satellite channel dispersion V_sat from WAN latency data...")
    
    # Filter RTT bins above 200ms (0.20 seconds) representing satellite backhaul
    sat_flows = wan_df[wan_df['rtt_bin_sec'] >= 0.20].copy()
    
    if sat_flows.empty:
        print("No flows found above 200ms to estimate satellite dispersion.")
        return 0.20, 0.0
        
    x = sat_flows['rtt_bin_sec'].values
    w = sat_flows['flow_count'].values
    
    total_flows = w.sum()
    if total_flows == 0:
         return 0.20, 0.0
         
    # Weighted mean
    mean_rtt = np.sum(w * x) / total_flows
    # Weighted variance (dispersion)
    var_rtt = np.sum(w * (x - mean_rtt)**2) / total_flows
    
    print(f"Satellite WAN Link Latency Summary (RTT >= 200ms):")
    print(f"  - Mean Satellite RTT: {mean_rtt*1000:.2f} ms")
    print(f"  - Empirical Dispersion V_sat: {var_rtt:.6f} sec^2")
    
    # Plotting: Latency distribution showing the satellite wall
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(wan_df['rtt_bin_sec'] * 1000, wan_df['flow_count'], color='#f39c12', alpha=0.7, edgecolor='black', width=10)
    ax.axvline(200, color='red', linestyle='--', linewidth=1.5, label='Satellite Wall (200ms)')
    
    ax.set_title("WAN Latency Distribution: Exposing the Satellite Wall", fontweight='bold', pad=15)
    ax.set_xlabel("Round Trip Time (RTT) (ms)")
    ax.set_ylabel("Flow Count")
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Text annotation
    ax.text(0.55, 0.75, f"Mean Sat RTT: {mean_rtt*1000:.1f} ms\nDispersion V_sat: {var_rtt:.6f} s^2", 
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='grey'))
            
    plt.tight_layout()
    plot_path = os.path.join(access.OUTPUT_PLOTS_FOLDER, "satellite_fbl_dispersion.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print("Generated satellite_fbl_dispersion.png")
    
    return mean_rtt, var_rtt

def estimate_core_signaling_aoi(gtpc_df):
    """
    Extracts core GTP-C request arrival rates and optimizes the Peak Age of Information
    (PAoI) over a finite M/GI/1/K queue.
    """
    print("Estimating core signaling parameters and Peak AoI optimization...")
    
    # Calculate empirical arrival rate from timestamps
    gtpc_df = gtpc_df.sort_values('timestamp_gmt')
    duration = (gtpc_df['timestamp_gmt'].max() - gtpc_df['timestamp_gmt'].min()).total_seconds()
    num_requests = len(gtpc_df)
    
    if duration == 0:
        duration = 1.0
        
    emp_lambda = num_requests / duration
    print(f"GTP-C Signaling Core Statistics:")
    print(f"  - Total Messages: {num_requests}")
    print(f"  - Trace Duration: {duration:.2f} seconds")
    print(f"  - Empirical Arrival Rate (lambda_emp): {emp_lambda:.2f} msgs/sec")
    
    # Set M/GI/1/K parameters
    # Assume typical core router hardware constraints:
    # Mean service time E[S] = 5ms (service rate mu = 200 msgs/sec)
    mu = 200.0
    e_s = 1.0 / mu
    e_s2 = 1.2 * (e_s**2) # E[S^2] with small variance (General service time distribution)
    K = 50 # Queue capacity
    
    # Vary lambda to plot the AoI optimization curve
    lambdas = np.linspace(1.0, 195.0, 100)
    aoi_vals = []
    drop_probs = []
    
    for l in lambdas:
        rho = l * e_s
        if np.abs(rho - 1.0) < 1e-5:
            p_drop = 1.0 / (K + 1)
        else:
            p_drop = ((1.0 - rho) * (rho**K)) / (1.0 - (rho**(K+1)))
            
        l_eff = l * (1.0 - p_drop)
        # Pollaczek-Khinchine proxy for waiting time
        w_time = (l_eff * e_s2) / (2.0 * (1.0 - l_eff * e_s + 1e-12))
        sys_time = w_time + e_s
        
        # Peak AoI: E[T] + 1 / (l * (1 - p_drop))
        peak_aoi = sys_time + (1.0 / (l * (1.0 - p_drop) + 1e-12))
        aoi_vals.append(peak_aoi * 1000) # Convert to ms
        drop_probs.append(p_drop * 100)
        
    # Locate optimal lambda*
    opt_idx = np.argmin(aoi_vals)
    opt_lambda = lambdas[opt_idx]
    opt_aoi = aoi_vals[opt_idx]
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    # Age of Information Curve
    ax1.plot(lambdas, aoi_vals, color='#2c3e50', linewidth=2.5, label='Peak AoI (ms)')
    ax1.axvline(opt_lambda, color='#2ecc71', linestyle='--', linewidth=1.5, 
                label=f'Optimal Rate lambda*: {opt_lambda:.1f} msgs/s')
    ax1.axvline(emp_lambda, color='#e74c3c', linestyle=':', linewidth=2.0, 
                label=f'Empirical Rate lambda_emp: {emp_lambda:.1f} msgs/s')
    
    ax1.set_xlabel("Signaling Arrival Rate lambda (messages/second)")
    ax1.set_ylabel("Peak Age of Information (ms)", color='#2c3e50')
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Drop Probability Curve (on secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(lambdas, drop_probs, color='#95a5a6', linestyle='-.', alpha=0.7, label='Drop Prob (%)')
    ax2.set_ylabel("Signaling Drop Probability (%)", color='#7f8c8d')
    ax2.tick_params(axis='y', labelcolor='#7f8c8d')
    ax2.spines['top'].set_visible(False)
    
    # Align legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    plt.title("Age of Information (AoI) Optimization under Core Signaling Load", fontweight='bold', pad=15)
    plt.tight_layout()
    
    plot_path = os.path.join(access.OUTPUT_PLOTS_FOLDER, "core_queue_aoi_optimization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated core_queue_aoi_optimization.png (Optimal lambda* = {opt_lambda:.2f} msgs/sec)")
    
    return opt_lambda, opt_aoi
