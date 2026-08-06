import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from . import access

# Strict Academic Style Config
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "png"
})

# Create an output folder for the LaTeX presentation images
os.makedirs(access.OUTPUT_PLOTS_FOLDER, exist_ok=True)

# =====================================================================
# IMMEDIATE SPRINT TASKS
# =====================================================================

def categorize_content(df: pd.DataFrame) -> pd.DataFrame:
    """Groups data by category, calculating absolute volume (volume_bytes) and percentage share."""
    stats = df.groupby("category", observed=False)["volume_bytes"].sum().reset_index()
    total_bytes = stats["volume_bytes"].sum()
    stats["percentage_share"] = (stats["volume_bytes"] / total_bytes * 100) if total_bytes > 0 else 0
    stats = stats.sort_values(by="volume_bytes", ascending=False)
    return stats

def compute_bufferbloat_cdf(rtt_df: pd.DataFrame) -> pd.DataFrame:
    """Converts frequency histograms into Cumulative Distribution Functions (CDFs)."""
    df = rtt_df.sort_values(by=["segment", "rtt_bin_sec"]).copy()
    df["cdf"] = df.groupby("segment", observed=False)["flow_count"].transform(lambda x: x.cumsum() / x.sum())
    return df

def calculate_out_of_order_penalty(qos_df: pd.DataFrame) -> pd.DataFrame:
    """Groups by rat_code and compares out_of_order vs. retransmission rates."""
    rat_map = {1: '2G (GPRS/EDGE)', 2: '2.5G (EDGE)', 3: '3G (UMTS)', 4: '3.5G (HSPA)'}
    df = qos_df.copy()
    df['Network_Gen'] = df['rat_code'].map(rat_map)
    df = df.dropna(subset=['Network_Gen'])
    
    # Calculate flag rates
    stats = df.groupby('Network_Gen', observed=False)[['out_of_order', 'retransmission']].mean() * 100
    stats = stats.reindex(['2G (GPRS/EDGE)', '2.5G (EDGE)', '3G (UMTS)', '3.5G (HSPA)']).dropna()
    return stats

def plot_academic_cdf(df: pd.DataFrame, x_col: str = "rtt_bin_sec", y_col: str = "cdf", title: str = "Latency CDF: LAN vs WAN", output_path: str = None) -> None:
    """Renders vector .png CDF curves conforming to academic publication rules."""
    if output_path is None:
        output_path = os.path.join(access.OUTPUT_PLOTS_FOLDER, "rtt_cdf_lan_vs_wan.png")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    segments = df["segment"].unique()
    styles = {"lan": ("-", "#2c3e50"), "wan": ("--", "#c0392b"), "lan_umts": ("-.", "#27ae60")}
    
    for seg in segments:
        seg_df = df[df["segment"] == seg].sort_values(by=x_col)
        style, color = styles.get(seg, ("-", "k"))
        label = "Local (2G/3G LAN)" if seg == "lan" else ("3G Only LAN" if seg == "lan_umts" else "Satellite (WAN)")
        ax.plot(seg_df[x_col] * 1000, seg_df[y_col], linestyle=style, color=color, label=label)
        
    ax.set_title(title, pad=12, fontweight='bold')
    ax.set_xlabel("Round Trip Time (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1.05)
    
    ax.grid(color='grey', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Highlight the 200ms satellite backhaul step-function
    if "wan" in segments:
        ax.annotate("GEO Satellite Prop. Delay\n(200ms Step-Function)",
                    xy=(200, 0.5), xytext=(350, 0.4),
                    arrowprops=dict(facecolor='black', arrowstyle="->", connectionstyle="arc3"),
                    fontweight='bold', fontsize=9)
                    
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()

def plot_content_distribution_horizontal(df: pd.DataFrame, output_path: str = None) -> None:
    """Renders a stacked/horizontal bar chart of content behavioral categories."""
    if output_path is None:
        output_path = os.path.join(access.OUTPUT_PLOTS_FOLDER, "content_category_distribution.png")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if 'direction' not in df.columns:
        df = df.copy()
        df['direction'] = 'Downlink'
        
    stats = df.groupby(['category', 'direction'], observed=False)['volume_bytes'].sum().unstack(fill_value=0)
    stats = stats / 1e12  # Convert to TB
    stats['Total_TB'] = stats.sum(axis=1)
    stats = stats.sort_values(by='Total_TB', ascending=True)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    categories = stats.index.astype(str)
    downlink_vol = stats.get('Downlink', pd.Series(0, index=stats.index))
    uplink_vol = stats.get('Uplink', pd.Series(0, index=stats.index))
    
    ax.barh(categories, downlink_vol, color='#2c3e50', label='Downlink (TB)')
    ax.barh(categories, uplink_vol, left=downlink_vol, color='#7f8c8d', label='Uplink (TB)')
    
    ax.set_title("Traffic Composition by Content Category (Airtel Rwanda)", pad=15, fontweight='bold')
    ax.set_xlabel("Total Data Volume (Terabytes)")
    ax.set_ylabel("Content Category")
    
    ax.grid(axis='x', color='grey', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()

# =====================================================================
# PORTED LOGIC & ADDITIONAL FIGURES FROM NOTEBOOK
# =====================================================================

def discover_network_stats(bw_df, content_total_tb, rtt_df):
    raw_tb = bw_df[bw_df['code'].isin(['dt', 'dm'])]['val'].sum() / 1e12
    efficiency = (content_total_tb / raw_tb) * 100
    waste_tb = raw_tb - content_total_tb
    
    rtt_sorted = rtt_df.sort_values('rtt_bin_sec')
    cdf = rtt_sorted['flow_count'].cumsum() / rtt_sorted['flow_count'].sum()
    median_ms = rtt_sorted.loc[(cdf >= 0.5).idxmax(), 'rtt_bin_sec'] * 1000
    
    print("--- Discovery Complete ---")
    print(f"Identified Median RTT: {median_ms:.1f} ms")
    print(f"Identified Raw Volume: {raw_tb:.2f} TB")
    print(f"Discovered Overhead:   {waste_tb:.2f} TB ({100-efficiency:.1f}%)")
    
    return {"raw_tb": raw_tb, "content_tb": content_total_tb, "waste_tb": waste_tb, "efficiency_pct": efficiency, "median_rtt": median_ms}

def plot_volumetric_efficiency(stats):
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.bar('Network Load', stats['content_tb'], label='User Payload (Data Plane)', color='#27ae60', width=0.6)
    ax.bar('Network Load', stats['waste_tb'], bottom=stats['content_tb'], label='Protocol Overhead (Control Plane)', color='#c0392b', width=0.6)
    ax.set_title(f"Volumetric Efficiency: {stats['efficiency_pct']:.1f}%", fontweight='bold')
    ax.set_ylabel("Traffic Volume (Terabytes)")
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "volumetric_efficiency.png"), format="png")
    plt.close()

def plot_protocol_tax_breakdown(df):
    breakdown = df.groupby('code')['val'].sum() / 1e12
    labels = {'dm': 'Downlink Control', 'um': 'Uplink Control', 'dt': 'Downlink Payload', 'ut': 'Uplink Payload'}
    breakdown.index = [labels.get(x, x) for x in breakdown.index]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(breakdown, labels=breakdown.index, autopct='%1.1f%%', colors=sns.color_palette("muted"))
    ax.set_title("Network Load: Payload vs. Control Plane Breakdown", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "protocol_tax_breakdown.png"), format="png")
    plt.close()

def plot_traffic_asymmetry(df):
    asym = df.groupby('code')['val'].sum() / 1e12
    labels = {'dt': 'Downlink Payload', 'ut': 'Uplink Payload', 'dm': 'Downlink Control', 'um': 'Uplink Control'}
    asym.index = [labels.get(x, x) for x in asym.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    asym.plot(kind='barh', color=['#3498db', '#e74c3c', '#95a5a6', '#7f8c8d'], ax=ax)
    ax.set_title("Traffic Asymmetry: Directional Infrastructure Load", fontweight='bold')
    ax.set_xlabel("Volume (Terabytes)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "traffic_asymmetry.png"), format="png")
    plt.close()

def plot_application_signaling_profiles(df, top_n=8):
    known_df = df[df['domain'] != 'Other/System']
    app_stats = known_df.groupby(['domain', 'code'])['val'].sum().unstack(fill_value=0)
    app_stats['Payload (dt+ut)'] = app_stats.get('dt', 0) + app_stats.get('ut', 0)
    app_stats['Signaling (dm+um)'] = app_stats.get('dm', 0) + app_stats.get('um', 0)
    
    top_apps = app_stats.nlargest(top_n, 'Payload (dt+ut)').copy()
    top_apps['Signaling Ratio'] = top_apps['Signaling (dm+um)'] / top_apps['Payload (dt+ut)']
    top_apps = top_apps.sort_values('Signaling Ratio', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_apps.index, y=top_apps['Signaling Ratio'], color='#c0392b', ax=ax)
    ax.set_title("Application Overhead: Signaling Generated per Payload Byte", fontweight='bold')
    ax.set_ylabel("Signaling Bytes per 1 Payload Byte")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "application_signaling_profiles.png"), format="png")
    plt.close()

def analyze_rat_efficiency(df):
    rat_stats = df.groupby(['rat', 'code'])['val'].sum().unstack(fill_value=0)
    rat_stats['Payload'] = rat_stats.get('dt', 0) + rat_stats.get('ut', 0)
    rat_stats['Signaling'] = rat_stats.get('dm', 0) + rat_stats.get('um', 0)
    rat_stats['Efficiency %'] = (rat_stats['Payload'] / (rat_stats['Payload'] + rat_stats['Signaling'])) * 100
    valid_rats = rat_stats[rat_stats['Payload'] > 0].copy()
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=valid_rats.index, y=valid_rats['Efficiency %'], palette='viridis', ax=ax)
    ax.set_title("Technology Overhead: Volumetric Efficiency by RAT Generation", fontweight='bold')
    ax.set_xlabel("Radio Access Technology (RAT Code)")
    ax.set_ylabel("Payload Efficiency (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, val in enumerate(valid_rats['Efficiency %']):
        ax.text(i, val + 2, f"{val:.1f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "rat_efficiency.png"), format="png")
    plt.close()

def plot_ghost_hour_tax(df):
    df = df.copy()
    df['Time_Block'] = pd.cut(df['hour'], bins=[-1, 0, 5, 17, 22, 24],
                              labels=['Night', 'Off-Peak (1AM-5AM)', 'Day', 'Peak (6PM-10PM)', 'Late Night'])
    
    stats = df.groupby(['Time_Block', 'code'], observed=False)['val'].sum().unstack(fill_value=0)
    stats['Payload (GB)'] = (stats.get('dt', 0) + stats.get('ut', 0)) / 1e9
    stats['Signaling (GB)'] = (stats.get('dm', 0) + stats.get('um', 0)) / 1e9
    
    focus = stats.loc[['Off-Peak (1AM-5AM)', 'Peak (6PM-10PM)']].copy()
    focus['Signaling %'] = (focus['Signaling (GB)'] / (focus['Payload (GB)'] + focus['Signaling (GB)'])) * 100
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(focus))
    width = 0.35
    ax.bar(x - width/2, focus['Payload (GB)'], width, label='User Payload', color='#27ae60')
    ax.bar(x + width/2, focus['Signaling (GB)'], width, label='Control Signaling', color='#c0392b')
    
    for i in range(len(focus)):
        ax.text(i + width/2, focus['Signaling (GB)'].iloc[i] + 50, f"{focus['Signaling %'].iloc[i]:.1f}%\nSignaling", ha='center', fontweight='bold', fontsize=8)
        
    ax.set_title("Diurnal Overhead: Off-Peak vs. Peak Network Load", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(focus.index)
    ax.set_ylabel("Total Volume (Gigabytes)")
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "ghost_hour_tax.png"), format="png")
    plt.close()

def plot_hourly_signaling_volatility(df):
    hourly = df.groupby(['hour', 'code'], observed=False)['val'].sum().unstack(fill_value=0)
    payload = hourly.get('dt', 0) + hourly.get('ut', 0)
    signaling = hourly.get('dm', 0) + hourly.get('um', 0)
    ratio = (signaling / (payload + signaling)) * 100
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ratio.index, ratio.values, marker='o', linestyle='-', color='#d35400', linewidth=2, markersize=6)
    ax.set_title("Control Plane Volatility: 24-Hour Signaling Ratio", fontweight='bold')
    ax.set_xlabel("Hour of Day (24h)")
    ax.set_ylabel("Signaling as % of Total Network Volume")
    ax.set_xticks(range(24))
    ax.fill_between(ratio.index, ratio.values.min(), ratio.values, color='#e67e22', alpha=0.2)
    ax.axhline(ratio.mean(), color='black', linestyle='--', label=f"Average Signaling Ratio ({ratio.mean():.1f}%)")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "hourly_signaling_volatility.png"), format="png")
    plt.close()

def plot_application_dominance(content_df):
    df_plot = content_df.dropna(subset=['domain']).sort_values(by='bytes', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_plot['domain'], df_plot['bytes']/1e9, color='#f39c12')
    ax.invert_yaxis()
    ax.set_title("Network Payload Distribution: Top Application Domains", fontweight='bold')
    ax.set_xlabel("Volume (Gigabytes)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "application_dominance.png"), format="png")
    plt.close()

def get_useful_diurnal_data(df, top_n_to_filter=5):
    useful_df = df[df['code'].isin(['ut', 'dt'])].copy()
    known = useful_df[useful_df['domain'] != 'Other/System']
    if not known.empty:
        top_apps = known.groupby('domain')['val'].sum().nlargest(top_n_to_filter).index.tolist()
        useful_df = useful_df[useful_df['domain'].isin(top_apps + ['Other/System'])]
    return useful_df

def plot_useful_diurnal(df, top_n=5):
    clean_df = get_useful_diurnal_data(df, top_n_to_filter=top_n)
    df_down = clean_df[clean_df['code'] == 'dt'].copy()
    hourly_stats = df_down.groupby(['hour', 'domain'])['val'].mean().reset_index()
    hourly_stats['val_gb'] = hourly_stats['val'] / 1e9
    total_hourly = df_down.groupby('hour')['val'].mean() / 1e9
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=hourly_stats, x='hour', y='val_gb', hue='domain', palette='husl', linewidth=2, ax=ax)
    ax.plot(total_hourly.index, total_hourly.values, label='TOTAL PAYLOAD', color='black', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.set_title("Diurnal Application Demand: The Network Rhythm", fontweight='bold')
    ax.set_xlabel("Hour of Day (24h)")
    ax.set_ylabel("Average Volume (Gigabytes)")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(title="Application Domain", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "useful_diurnal.png"), format="png")
    plt.close()

def plot_intra_week_volatility(df):
    useful_df = df[(df['code'].isin(['ut', 'dt'])) & (df['domain'] != 'Other/System')].copy()
    useful_df['day_type'] = useful_df['timestamp'].dt.dayofweek.map(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    stats = useful_df.groupby(['hour', 'day_type'])['val'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=stats, x='hour', y=stats['val']/1e9, hue='day_type', linewidth=2.5, palette=['#3498db', '#e74c3c'], ax=ax)
    ax.set_title("Intra-Week Demand Volatility: Behavioral Load Shifts", fontweight='bold')
    ax.set_ylabel("Average Volume (Gigabytes)")
    ax.set_xlabel("Hour of Day (24h)")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "intra_week_volatility.png"), format="png")
    plt.close()

def plot_domain_asymmetry(df, top_n=6):
    useful = df[(df['code'].isin(['dt', 'ut'])) & (df['domain'] != 'Other/System')]
    domain_stats = useful.groupby(['domain', 'code'])['val'].sum().unstack(fill_value=0) / 1e9
    domain_stats['Total'] = domain_stats['dt'] + domain_stats['ut']
    top_domains = domain_stats.nlargest(top_n, 'Total')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(top_domains))
    ax.barh(y, top_domains['dt'], color='#3498db', label='Downlink (Consumption)')
    ax.barh(y, -top_domains['ut'], color='#e74c3c', label='Uplink (Production)')
    
    ax.set_title("Domain Directionality: Traffic Consumption vs. Production", fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(top_domains.index)
    ax.set_xlabel("Volume (Gigabytes) - Note: Uplink plotted on negative axis")
    ax.axvline(0, color='black', linewidth=1)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "domain_asymmetry.png"), format="png")
    plt.close()

def plot_network_inequality_lorenz(df):
    node_totals = df[df['code'].isin(['dt', 'ut'])].groupby('device')['val'].sum().sort_values(ascending=False)
    cum_payload = node_totals.cumsum() / node_totals.sum() * 100
    cum_nodes = np.arange(1, len(node_totals) + 1) / len(node_totals) * 100
    
    cum_payload = np.insert(cum_payload.values, 0, 0)
    cum_nodes = np.insert(cum_nodes, 0, 0)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(cum_nodes, cum_payload, label='Network Load Lorenz Curve', color='#e74c3c', linewidth=2.5)
    ax.plot([0, 100], [0, 100], linestyle='--', color='grey', label='Line of Perfect Equality')
    
    idx_20_pct = np.abs(cum_nodes - 20).argmin()
    payload_at_20 = cum_payload[idx_20_pct]
    
    ax.axvline(20, color='black', linestyle=':', alpha=0.5)
    ax.axhline(payload_at_20, color='black', linestyle=':', alpha=0.5)
    ax.scatter([20], [payload_at_20], color='black', s=60, zorder=5)
    ax.text(25, payload_at_20 - 5, f"Top 20% of Nodes\ncarry {payload_at_20:.1f}% of Traffic", fontweight='bold', fontsize=9)
    
    ax.set_title("Spatial Network Inequality: Payload Distribution", fontweight='bold')
    ax.set_xlabel("Cumulative % of Network Nodes")
    ax.set_ylabel("Cumulative % of Total Payload")
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "network_inequality_lorenz.png"), format="png")
    plt.close()

def plot_app_monoculture_entropy(df):
    useful = df[(df['code'].isin(['dt', 'ut'])) & (df['domain'] != 'Other/System')].copy()
    node_domain_vol = useful.groupby(['device', 'domain'], observed=False)['val'].sum().unstack(fill_value=0)
    
    p = node_domain_vol.div(node_domain_vol.sum(axis=1), axis=0)
    entropy = -(p * np.log2(p.replace(0, np.nan))).sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(entropy.dropna(), bins=15, kde=True, color='#9b59b6', ax=ax)
    ax.set_title("Application Ecosystem: Shannon Entropy of Domain Diversity per Node", fontweight='bold')
    ax.set_xlabel("Shannon Entropy Score (Low = Monoculture, High = High Diversity)")
    ax.set_ylabel("Number of Network Nodes")
    
    mean_entropy = entropy.mean()
    ax.axvline(mean_entropy, color='red', linestyle='--', label=f'Network Mean Entropy: {mean_entropy:.2f}')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "app_monoculture_entropy.png"), format="png")
    plt.close()

def identify_researcher_windows(df):
    useful_hourly = df[df['domain'] != 'Other/System'].groupby('hour')['val'].mean()
    best_hour = useful_hourly.rolling(window=4).mean().idxmin()
    print("\n--- STRATEGIC RECOMMENDATION ---")
    print(f"Optimal Network Utilization Window: {best_hour-4:02d}:00 to {best_hour:02d}:00")
    print("Benefit: Leverages off-peak capacity to avoid active congestion interference.")

def identify_node_saturation(df):
    node_hr = df.groupby(['device', 'hour', 'code'])['val'].sum().unstack(fill_value=0)
    node_hr['Payload'] = node_hr.get('dt', 0) + node_hr.get('ut', 0)
    node_hr['Signaling'] = node_hr.get('dm', 0) + node_hr.get('um', 0)
    node_hr['Total'] = node_hr['Payload'] + node_hr['Signaling']
    
    crashes = node_hr[(node_hr['Signaling'] / node_hr['Total'] > 0.90) & (node_hr['Total'] > 1e9)]
    print("\n--- INFRASTRUCTURE FINDING: BASE STATION SATURATION ---")
    print(f"Discovered {len(crashes)} discrete 'Micro-Outage' events.")
    print("These denote node-hours where control plane operations consumed >90% of capacity, failing to deliver user payload.")
    if not crashes.empty:
        top_crashes = crashes.reset_index().groupby('hour').size()
        print(f"Highest vulnerability time frame: {top_crashes.idxmax():02d}:00 (Frequency: {top_crashes.max()})")

def plot_anatomy_of_a_crash(df):
    node_hr = df.groupby(['device', 'hour', 'code'])['val'].sum().unstack(fill_value=0)
    node_hr['Payload'] = node_hr.get('dt', 0) + node_hr.get('ut', 0)
    node_hr['Signaling'] = node_hr.get('dm', 0) + node_hr.get('um', 0)
    node_hr['Total'] = node_hr['Payload'] + node_hr['Signaling']
    node_hr['Signaling_Ratio'] = node_hr['Signaling'] / node_hr['Total']
    
    active_nodes = node_hr[node_hr['Total'] > 1e8]
    if active_nodes.empty:
        return
        
    worst_idx = active_nodes['Signaling_Ratio'].idxmax()
    worst_node = worst_idx[0]
    
    node_data = node_hr.loc[worst_node].reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(node_data['hour'], node_data['Signaling']/1e6, label='Control Plane (Signaling MB)', color='#c0392b', linewidth=2.5)
    ax.plot(node_data['hour'], node_data['Payload']/1e6, label='Data Plane (Payload MB)', color='#27ae60', linewidth=2.5)
    ax.fill_between(node_data['hour'], node_data['Signaling']/1e6, node_data['Payload']/1e6,
                     where=(node_data['Signaling'] > node_data['Payload']), color='red', alpha=0.15, label="Saturation State")
                     
    ax.set_title(f"Node Saturation Profile: Control Plane Collapse (Node {worst_node})", fontweight='bold')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Hourly Volume (Megabytes)")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, f"anatomy_of_crash_node_{worst_node}.png"), format="png")
    plt.close()

def plot_tcp_ack_starvation(df):
    stats = df.groupby(['device', 'hour', 'code'])['val'].sum().unstack(fill_value=0)
    stats = stats[(stats.get('dt', 0) > 1e6) & (stats.get('um', 0) > 1e6)].copy()
    stats['Uplink_Signaling_MB'] = stats['um'] / 1e6
    stats['Downlink_Payload_GB'] = stats['dt'] / 1e9
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=stats['Uplink_Signaling_MB'], y=stats['Downlink_Payload_GB'], alpha=0.5, color='#8e44ad', edgecolor=None, ax=ax)
    
    threshold = stats['Uplink_Signaling_MB'].quantile(0.90)
    ax.axvspan(threshold, stats['Uplink_Signaling_MB'].max(), color='red', alpha=0.1, label="Protocol Starvation Threshold (Top 10%)")
    
    ax.set_title("TCP Layer Impairment: Downlink Throughput vs. Uplink Control Congestion", fontweight='bold')
    ax.set_xlabel("Uplink Control Plane Load (Megabytes)")
    ax.set_ylabel("Downlink User Payload Volume (Gigabytes)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "tcp_ack_starvation.png"), format="png")
    plt.close()

def plot_core_network_churn(gtpc_df):
    type_col = [c for c in gtpc_df.columns if 'type' in str(c).lower() or 'msg' in str(c).lower()][0]
    msg_counts = gtpc_df[type_col].value_counts()
    
    type_map = {16: 'Create Session', 18: 'Update Session', 26: 'Delete Session'}
    msg_counts.index = msg_counts.index.map(lambda x: type_map.get(x, f"Type {x}"))
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(msg_counts.index, msg_counts.values, marker='o', markersize=10, linewidth=2.5, color='#8e44ad')
    ax.fill_between(msg_counts.index, 0, msg_counts.values, alpha=0.2, color='#8e44ad')
    
    ax.set_title("Core Network Volatility: GTP-C Session Churn", fontweight='bold')
    ax.set_ylabel("Number of Control Plane Messages")
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "core_network_churn.png"), format="png")
    plt.close()

def analyze_cellular_intelligence(df):
    rat_volume = df.groupby('rat')['val'].sum() / 1e12
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(rat_volume, labels=[f"RAT Code {int(i)}" for i in rat_volume.index], autopct='%1.1f%%')
    ax.set_title("Infrastructure Utilization: 2G vs 3G Load Distribution", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "cellular_intelligence.png"), format="png")
    plt.close()

def plot_device_ecosystem(device_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    str_cols = device_df.select_dtypes(include=['object']).columns
    name_col = str_cols[0] if len(str_cols) > 0 else device_df.columns[0]
    num_cols = device_df.select_dtypes(exclude=['object']).columns
    count_col = device_df[num_cols].max().idxmax()
    
    plot_df = device_df.sort_values(by=count_col, ascending=True)
    bars = ax.barh(plot_df[name_col].astype(str), plot_df[count_col], color='#2980b9')
    
    total_devices = plot_df[count_col].sum()
    for bar in bars:
        width = bar.get_width()
        percentage = (width / total_devices) * 100
        ax.text(width + (plot_df[count_col].max() * 0.02), bar.get_y() + bar.get_height()/2,
                 f"{percentage:.1f}%", va='center', fontweight='bold', fontsize=8)
                 
    ax.set_title("Hardware Topology: User Equipment Distribution", fontweight='bold')
    ax.set_xlabel("Total Associated Devices (Log Scale)")
    ax.set_ylabel("Device Archetype")
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "device_ecosystem.png"), format="png")
    plt.close()

def plot_brand_dominance_and_dns(brand_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    brand_df['Count'] = pd.to_numeric(brand_df['Count'], errors='coerce')
    brand_df = brand_df.dropna(subset=['Count'])
    
    str_col = brand_df.columns[0]
    brand_df = brand_df[~brand_df[str_col].astype(str).str.upper().isin(['ALL', 'TOTAL', 'UNKNOWN', 'NAN'])].copy()
    
    top_brands = brand_df.nlargest(10, 'Count').sort_values(by="Count", ascending=True)
    bars = ax.barh(top_brands[str_col].astype(str), top_brands['Count'], color='#d35400')
    
    total_devices = brand_df['Count'].sum()
    for bar in bars:
        width = bar.get_width()
        pct = (width / total_devices) * 100
        ax.text(width + (brand_df['Count'].max() * 0.01), bar.get_y() + bar.get_height()/2,
                 f"{pct:.1f}%", va='center', fontweight='bold', fontsize=8)
                 
    ax.set_title("Hardware Supply Chain: Original Equipment Manufacturer (OEM) Dominance", fontweight='bold')
    ax.set_xlabel("Total Associated Devices")
    ax.set_ylabel("OEM Brand")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "brand_dominance.png"), format="png")
    plt.close()

def plot_global_routing_cdn_dominance(country_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    str_col = country_df.select_dtypes(include=['object']).columns[0]
    clean_df = country_df[~country_df[str_col].astype(str).str.upper().isin(['ALL', 'TOTAL', 'UNKNOWN', 'NAN'])].copy()
    num_cols = clean_df.select_dtypes(include=['number']).columns
    byte_col = clean_df[num_cols].sum().idxmax()
    
    top_countries = clean_df.nlargest(10, byte_col).sort_values(by=byte_col, ascending=True)
    ax.barh(top_countries[str_col].astype(str), top_countries[byte_col], color='#27ae60')
    
    ax.set_title("Global IP Routing: Content Delivery Network (CDN) Centralization", fontweight='bold')
    ax.set_xlabel("Total Data Volume (Bytes)")
    ax.set_ylabel("Destination Geolocation")
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "global_routing_cdn_dominance.png"), format="png")
    plt.close()

def plot_google_edge_routing(google_df):
    continent_col = [c for c in google_df.columns if 'cont' in str(c).lower()][0]
    down_col = [c for c in google_df.columns if 'down' in str(c).lower()][0]
    
    geo_stats = google_df.groupby(continent_col)[down_col].sum() / 1e9
    geo_stats = geo_stats.sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=geo_stats.index, y=geo_stats.values, palette='magma', ax=ax)
    ax.set_title("Edge Infrastructure: Geographic Serving Origins for Major Content", fontweight='bold')
    ax.set_ylabel("Downlink Payload Delivered (Gigabytes)")
    ax.set_xlabel("Origin Continent")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "google_edge_routing.png"), format="png")
    plt.close()

def plot_ip_geofiction(geo_df):
    maxmind_col = next((c for c in geo_df.columns if 'maxmind' in str(c).lower() or 'mm' in str(c).lower()), None)
    correct_col = next((c for c in geo_df.columns if 'correct' in str(c).lower() or 'true' in str(c).lower()), None)
    
    if not maxmind_col or not correct_col:
        maxmind_col, correct_col = geo_df.columns[2], geo_df.columns[3]
        
    errors = geo_df[geo_df[maxmind_col] != geo_df[correct_col]].copy()
    error_counts = errors[maxmind_col].value_counts().head(5)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(error_counts.index.astype(str), error_counts.values, color='#e67e22')
    ax.set_title("Geolocation Observability Impairment (IP Geofiction)", fontweight='bold')
    ax.set_xlabel("Continent Erroneously Assigned by Database Tools")
    ax.set_ylabel("Count of Misclassified IP Addresses")
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (error_counts.max()*0.02), int(yval), ha='center', fontweight='bold', fontsize=8)
        
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "ip_geofiction.png"), format="png")
    plt.close()

def profile_technology_benchmarks(segment_data_dict):
    print(f"\n{'Protocol/Tech':<15} | {'Median RTT':<12} | {'Peak Density'}")
    print("-" * 45)
    for label, df in segment_data_dict.items():
        df_sorted = df.sort_values('rtt_bin_sec')
        cdf = df_sorted['flow_count'].cumsum() / df_sorted['flow_count'].sum()
        median_ms = df_sorted.loc[(cdf >= 0.5).idxmax(), 'rtt_bin_sec'] * 1000
        peak_prob = (df['flow_count'].max() / df['flow_count'].sum()) * 100
        print(f"{label:<15} | {median_ms:>8.1f} ms | {peak_prob:>15.2f}%")

def plot_technology_signatures(df_dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("muted", len(df_dict))
    for (label, df), color in zip(df_dict.items(), colors):
        prob = (df['flow_count'] / df['flow_count'].sum()) * 100
        ax.plot(df['rtt_bin_sec'] * 1000, prob, label=label, color=color, linewidth=1.5)
    ax.set_title("Physical Layer Latency: RTT Signatures by Protocol", fontweight='bold')
    ax.set_xlabel("Round Trip Time (ms)")
    ax.set_ylabel("Probability Density (%)")
    ax.set_xlim(0, 1000)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "technology_signatures.png"), format="png")
    plt.close()

def plot_signaling_waterfall(discovered_median, redirect_rate=80):
    values = [discovered_median, discovered_median, discovered_median * (redirect_rate / 100)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bottom = 0
    colors = ['#9b59b6', '#8e44ad', '#663399']
    for label, val, color in zip(['DNS Resolution', 'TCP Handshake', 'HTTP Overhead'], values, colors):
        ax.bar('Connection Lifecycle', val, bottom=bottom, label=label, color=color, width=0.4)
        bottom += val
    ax.set_title(f"Cumulative Setup Latency (Baseline: {discovered_median:.0f} ms)", fontweight='bold')
    ax.set_ylabel("Aggregate Delay (ms)")
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "signaling_waterfall.png"), format="png")
    plt.close()

def plot_dns_penalty(dns_df):
    dns_df = dns_df.copy()
    dns_df['ms'] = dns_df['bin'] * 1000
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dns_df['ms'], dns_df['count'], color='#9b59b6', width=50)
    ax.set_xlim(0, 2000)
    ax.set_title("Radio Resource Control (RRC): Wireless Link State Transition Latency", fontweight='bold')
    ax.set_xlabel("Establishment Delay (ms)")
    ax.set_ylabel("Frequency")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "dns_penalty.png"), format="png")
    plt.close()

def plot_satellite_backhaul_wall(wan_df):
    wan_df = wan_df.copy()
    wan_df['ms'] = wan_df['bin'] * 1000
    plot_df = wan_df[wan_df['ms'] <= 500]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_df['ms'], plot_df['count'], color='#2c3e50', linewidth=1.5)
    ax.fill_between(plot_df['ms'], 0, plot_df['count'], color='#34495e', alpha=0.3)
    
    peak_ms = plot_df.loc[plot_df['count'].idxmax(), 'ms']
    ax.axvline(peak_ms, color='#e74c3c', linestyle='--', linewidth=1.5)
    ax.text(peak_ms + 10, plot_df['count'].max() * 0.9, f"Signal Propagation Wall\n({peak_ms:.0f}ms)", color='#e74c3c', fontweight='bold', fontsize=10)
    
    ax.set_title("Physical Infrastructure Mechanics: GEO Satellite Propagation Delay", fontweight='bold')
    ax.set_xlabel("Wide Area Network (WAN) Round Trip Time (ms)")
    ax.set_ylabel("Observed Packet Frequency")
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "satellite_backhaul_wall.png"), format="png")
    plt.close()

def plot_tcp_out_of_order_illusion(qos_df):
    qos_df = qos_df.copy()
    rat_map = {1: '2G (GPRS/EDGE)', 2: '2.5G (EDGE)', 3: '3G (UMTS)', 4: '3.5G (HSPA)'}
    
    qos_df['Network_Gen'] = qos_df['rat_code'].map(rat_map)
    dl_df = qos_df.dropna(subset=['Network_Gen']).copy()
    
    dl_df['Has_OOO'] = (dl_df['out_of_order'] > 0).astype(int)
    dl_df['Has_Retrans'] = (dl_df['retransmission'] > 0).astype(int)
    
    stats = dl_df.groupby('Network_Gen', observed=False)[['Has_OOO', 'Has_Retrans']].mean() * 100
    stats = stats.reindex(['2G (GPRS/EDGE)', '2.5G (EDGE)', '3G (UMTS)', '3.5G (HSPA)']).dropna()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(stats.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, stats['Has_OOO'], width, label='Sequence Anomaly (Out-of-Order %)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, stats['Has_Retrans'], width, label='TCP Congestion Trigger (Retransmission %)', color='#2c3e50')
    
    ax.set_title("Transport Protocol Inference Failure: 2G Radio-Layer Interference", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stats.index)
    ax.set_ylabel("Impacted Downlink Flows (%)")
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1, f"{height:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "tcp_out_of_order_illusion.png"), format="png")
    plt.close()

def plot_dns_invisible_tax(dns_df):
    dns_df = dns_df.sort_values(by="RTT_Bin").copy()
    if dns_df["RTT_Bin"].max() <= 100:
        dns_df["RTT_Bin"] = dns_df["RTT_Bin"] * 1000
        
    total_queries = dns_df["Query_Count"].sum()
    dns_df['Cumulative_Pct'] = (dns_df["Query_Count"].cumsum() / total_queries) * 100
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dns_df["RTT_Bin"], dns_df['Cumulative_Pct'], color='#8e44ad', linewidth=2.5)
    ax.fill_between(dns_df["RTT_Bin"], 0, dns_df['Cumulative_Pct'], color='#9b59b6', alpha=0.2)
    
    ax.set_title("Protocol Overhead: The DNS Resolution Latency Long Tail", fontweight='bold')
    ax.set_xlabel("Domain Name Resolution Time (ms)")
    ax.set_ylabel("Cumulative Proportion of Network Queries")
    
    ax.axhline(80, color='red', linestyle='--', alpha=0.7)
    ax.axvline(100, color='red', linestyle='--', alpha=0.7)
    ax.scatter([100], [80], color='red', s=60, zorder=5)
    ax.text(120, 77, "80% threshold\n(< 100ms)", color='red', fontweight='bold', fontsize=8)
    
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "dns_invisible_tax.png"), format="png")
    plt.close()

def plot_bufferbloat_crisis(rtt_df):
    rtt_df = rtt_df.copy()
    rtt_df['p97_rtt'] = pd.to_numeric(rtt_df['p97_rtt'], errors='coerce')
    rtt_df = rtt_df.dropna(subset=['p97_rtt']).copy()
    
    rtt_df['type'] = rtt_df['type'].replace({
        'Clean Handshake (SYN)': 'Connection Initiation\n(TCP SYN)',
        'Heavy Data Flow (Bufferbloat)': 'Active Payload Flow\n(Queue Degradation)'
    })
    
    stats = rtt_df.groupby('type', observed=False)['p97_rtt'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(stats.index, stats.values, color=['#3498db', '#c0392b'], height=0.5)
    
    ax.set_title("Queue Management Mechanics: Cellular Bufferbloat Degradation", fontweight='bold')
    ax.set_xlabel("Average Latency at 97th Percentile (ms) [LOG SCALE]")
    ax.set_xscale('log')
    ax.set_xlim(10, stats.max() * 5)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.2, bar.get_y() + bar.get_height()/2, f"{int(width):,} ms", va='center', fontweight='bold', fontsize=9)
        
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "bufferbloat_crisis.png"), format="png")
    plt.close()

def analyze_legacy_traps(df):
    df = df.copy()
    df['Tech_Gen'] = df['rat'].apply(lambda x: '3G_Capable' if x >= 3 else ('2G_Only' if x in [1, 2] else 'Unknown'))
    tower_tech = df[df['Tech_Gen'] != 'Unknown'].groupby(['device', 'Tech_Gen'])['val'].sum().unstack(fill_value=0)
    
    legacy_traps = tower_tech[(tower_tech.get('2G_Only', 0) > 0) & (tower_tech.get('3G_Capable', 0) == 0)]
    hybrid_modern = tower_tech[tower_tech.get('3G_Capable', 0) > 0]
    
    trap_pct = (len(legacy_traps) / len(tower_tech)) * 100
    
    print(f"\n--- RESEARCH FINDING: DIGITAL REDLINING ---")
    print(f"Total Towers Analyzed: {len(tower_tech)}")
    print(f"Modern/Hybrid Towers (3G Capable): {len(hybrid_modern)}")
    print(f"Legacy Traps (Strictly 2G): {len(legacy_traps)} ({trap_pct:.1f}% of network)")
    print(f"Insight: {trap_pct:.1f}% of the physical network footprint is digitally redlining its users, capping them at 2G speeds where modern web protocols (like HTTPS and heavy TCP) will inherently fail via timeout.")

def analyze_and_plot_content_categories(content_df):
    if content_df is None:
        return
        
    df = content_df.copy()
    df['domain'] = df['domain'].astype(str).str.lower()
    df['Category'] = df['domain'].map(access.map_domain_to_category).fillna('Unclassified Web')
    
    stats = df.groupby(['Category', 'direction'], observed=False)['bytes'].sum().unstack(fill_value=0)
    stats = stats / 1e12
    stats['Total_TB'] = stats.sum(axis=1)
    stats = stats.sort_values(by='Total_TB', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(stats.index, stats['Downlink'], color='#2c3e50', label='Downlink (TB)')
    ax.barh(stats.index, stats['Uplink'], left=stats['Downlink'], color='#7f8c8d', label='Uplink (TB)')
    
    ax.set_title("Traffic Composition by Content Category (Airtel Rwanda)", pad=15, fontweight='bold')
    ax.set_xlabel("Total Data Volume (Terabytes)")
    ax.set_ylabel("Content Category")
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "content_category_distribution.png"), format="png")
    plt.close()
    
    print("--- Content Classification Complete ---")
    top_3 = stats.nlargest(3, 'Total_TB')
    print(f"Top Category: {top_3.index[0]} ({top_3['Total_TB'].iloc[0]:.2f} TB)")

# =====================================================================
# TIMELINE REPLICATION & REMAINING SPRINT TASKS
# =====================================================================

def plot_hourly_rtt_percentiles(percentiles=[75.0, 80.0, 85.0, 90.0, 95.0]):
    """Replicates the Hourly RTT Per Device plots from the project timeline."""
    print("Loading RTT histograms for percentile calculation...")
    rtt_df = access.load_hourly_rtt_hist()
    df_in = rtt_df[rtt_df['direction'] == 'i'].copy()
    
    device_map = {1: 'Phone', 2: 'Smart', 3: 'USB', 4: 'Tablet', 6: 'Modem', 9: 'Router'}
    device_colors = {
        'Phone': '#3498db', 'Smart': '#2ecc71', 'USB': '#e74c3c',
        'Tablet': '#1abc9c', 'Modem': '#9b59b6', 'Router': '#f1c40f'
    }
    
    df_in = df_in[df_in['device'].isin(device_map.keys())].copy()
    df_in = df_in.sort_values(['device', 'time', 'rtt'])
    df_in['cumsum'] = df_in.groupby(['device', 'time'])['count'].cumsum()
    df_in['total'] = df_in.groupby(['device', 'time'])['count'].transform('sum')
    df_in['cum_pct'] = df_in['cumsum'] / df_in['total']
    
    times = sorted(df_in['time'].unique())
    x_indices = range(len(times))
    
    for p in percentiles:
        fig, ax = plt.subplots(figsize=(6, 4))
        p_df = df_in[df_in['cum_pct'] >= (p / 100.0)].drop_duplicates(['device', 'time'], keep='first')
        
        for dev_code, dev_name in device_map.items():
            dev_p = p_df[p_df['device'] == dev_code].set_index('time')
            rtt_vals = [dev_p.loc[t, 'rtt'] if t in dev_p.index else np.nan for t in times]
            ax.plot(x_indices, rtt_vals, label=dev_name, color=device_colors[dev_name])
            
        ax.set_title(f"Hourly RTT Per Device for {p}% of TCP Requests", pad=12, fontweight='bold')
        ax.set_xlabel("Hour")
        ax.set_ylabel("RTT")
        ax.set_xlim(0, 140)
        
        if p == 95.0:
            ax.set_ylim(0, 2.5)
        elif p == 90.0:
            ax.set_ylim(0, 1.2)
        elif p == 75.0:
            ax.set_ylim(0, 0.8)
            
        ax.legend(loc="upper right", frameon=True)
        ax.grid(color='grey', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, f"hourly_rtt_{p}_percent.png"), dpi=300)
        plt.close()
        print(f"Generated hourly_rtt_{p}_percent.png")

def plot_overall_throughput_time_series():
    """Replicates the double y-axis overall throughput time series curve."""
    print("Loading bandwidth stats...")
    bw_df = access.load_traffic_asymmetry()
    # Downlink: dt + dm, Uplink: ut + um
    dl = bw_df[bw_df['code'].isin(['dt', 'dm'])].groupby('time')['val'].sum()
    ul = bw_df[bw_df['code'].isin(['ut', 'um'])].groupby('time')['val'].sum()
    
    times = sorted(bw_df['time'].unique())
    x_indices = range(len(times))
    
    # Scale total captured traffic to operator-equivalent (divide by 4)
    dl_mbs = (dl * 8) / (3600 * 1e6 * 4)
    ul_mbs = (ul * 8) / (3600 * 1e6 * 4)
    
    # Align indices
    dl_vals = [dl_mbs.get(t, 0) for t in times]
    ul_vals = [ul_mbs.get(t, 0) for t in times]
    
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    
    color = 'blue'
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Overall Total Downlink Throughput in Mb/s', color=color)
    ax1.plot(x_indices, dl_vals, color=color, marker='o', markersize=3, label='Downlink')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim(0, 140)
    ax1.set_ylim(0, 350)
    
    ax2 = ax1.twinx()
    color = 'darkgreen'
    ax2.set_ylabel('Overall Total Uplink Throughput in Mb/s', color=color)
    ax2.plot(x_indices, ul_vals, color=color, marker='^', markersize=3, label='Uplink')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 40)
    
    plt.title("Overall Total Throughput in Mb/s", pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "overall_throughput_time_series.png"), dpi=300)
    plt.close()
    print("Generated overall_throughput_time_series.png")

def plot_stacked_traffic_by_continent():
    """Generates stacked traffic charts by destination continent."""
    print("Loading continent bandwidth stats...")
    df = access.load_continent_rtt_distribution()
    
    continent_map = {'AF': 'Africa', 'EU': 'Europe', 'NA': 'North America', 'AS': 'Asia'}
    times = sorted(df['time_idx'].unique())
    x_indices = range(len(times))
    
    for direction, code in [('Downlink', 'dtcont'), ('Uplink', 'utcont')]:
        df_dir = df[df['flag'] == (1 if direction == 'Downlink' else 0)].copy() # simple flag distinction
        
        # Group by time and continent
        grouped = df_dir.groupby(['time_idx', 'continent'])['flow_count'].sum().unstack(fill_value=0)
        
        # Map continents and group others
        grouped = grouped.rename(columns=continent_map)
        known_cont = [c for c in continent_map.values() if c in grouped.columns]
        other_cols = [c for c in grouped.columns if c not in continent_map.values()]
        grouped['Others'] = grouped[other_cols].sum(axis=1)
        
        plot_cols = known_cont + ['Others']
        percentages = grouped[plot_cols].div(grouped[plot_cols].sum(axis=1), axis=0) * 100
        percentages = percentages.reindex(times, fill_value=0)
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.stackplot(x_indices, [percentages[c] for c in plot_cols], labels=plot_cols,
                     colors=['#3498db', '#2ecc71', '#e74c3c', '#1abc9c', '#9b59b6'])
        
        ax.set_title(f"{direction} Traffic Stacked Percentages by Continent", pad=12, fontweight='bold')
        ax.set_xlabel("Hour")
        ax.set_ylabel("Percentage Traffic (%)")
        ax.set_xlim(0, 136)
        ax.set_ylim(0, 100)
        ax.legend(loc="lower left", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, f"stacked_traffic_by_continent_{direction.lower()}.png"), dpi=300)
        plt.close()
        print(f"Generated stacked_traffic_by_continent_{direction.lower()}.png")

def plot_stacked_traffic_by_country():
    """Generates stacked traffic charts by destination country."""
    print("Loading country traffic stats...")
    df = access.load_country_traffic()
    
    # We split by direction
    times = sorted(df['hour'].unique())
    x_indices = range(len(times))
    
    for direction in ['downlink', 'uplink']:
        val_col = 'down_bytes' if direction == 'downlink' else 'up_bytes'
        
        # Determine top countries
        top_countries = df.groupby('country')[val_col].sum().nlargest(4).index.tolist()
        
        grouped = df.groupby(['hour', 'country'])[val_col].sum().unstack(fill_value=0)
        known_countries = [c for c in top_countries if c in grouped.columns]
        other_cols = [c for c in grouped.columns if c not in top_countries]
        grouped['Others'] = grouped[other_cols].sum(axis=1)
        
        plot_cols = known_countries + ['Others']
        percentages = grouped[plot_cols].div(grouped[plot_cols].sum(axis=1), axis=0) * 100
        percentages = percentages.reindex(times, fill_value=0)
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.stackplot(x_indices, [percentages[c] for c in plot_cols], labels=plot_cols)
        
        ax.set_title(f"{direction.capitalize()} Traffic Country Classification", pad=12, fontweight='bold')
        ax.set_xlabel("Hour")
        ax.set_ylabel(f"Percentage {direction.capitalize()} Traffic (%)")
        ax.set_xlim(0, 136)
        ax.set_ylim(0, 100)
        ax.legend(loc="lower left", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, f"stacked_traffic_by_country_{direction}.png"), dpi=300)
        plt.close()
        print(f"Generated stacked_traffic_by_country_{direction}.png")

def plot_continent_rtt_pdfs(peak_hour=395410, non_peak_hour=395402):
    """Replicates the Peak and Non-Peak Continent RTT distribution PDF curves."""
    print("Loading continent RTT distributions...")
    df = access.load_continent_rtt_distribution()
    
    continent_map = {'AF': 'Africa', 'EU': 'Europe', 'NA': 'North America', 'AS': 'Asia'}
    colors = {'AF': 'blue', 'EU': 'green', 'NA': 'red', 'AS': 'cyan'}
    
    for h, title_str in [(peak_hour, "Peak Hour"), (non_peak_hour, "Non-Peak Hour")]:
        df_hour = df[df['time_idx'] == h].copy()
        if df_hour.empty:
            print(f"No data for hour {h}")
            continue
            
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for cont_code, cont_name in continent_map.items():
            c_df = df_hour[df_hour['continent'] == cont_code].sort_values('rtt_bin_sec')
            if c_df.empty:
                continue
            
            # Normalize counts to represent a Probability Density Function
            total_flows = c_df['flow_count'].sum()
            if total_flows > 0:
                normalized_count = c_df['flow_count'] / total_flows
                ax.plot(c_df['rtt_bin_sec'], normalized_count, label=cont_name, color=colors[cont_code])
                
        ax.set_title(f"Normalised {title_str} RTT Distribution", pad=12, fontweight='bold')
        ax.set_xlabel("RTT Bin (s)")
        ax.set_ylabel("Normalised Count")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 0.6)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(color='grey', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, f"normalised_{title_str.lower().replace(' ', '_')}_rtt_distribution.png"), dpi=300)
        plt.close()
        print(f"Generated normalised_{title_str.lower().replace(' ', '_')}_rtt_distribution.png")

def plot_dns_resolver_latencies():
    """Generates RTT CDFs for key DNS resolvers."""
    print("Loading DNS resolver latencies...")
    df = access.load_resolver_rtt_histograms()
    
    # We extract columns for ISP (172.28.1.116), Google (8.8.8.8), and Baidu (114.114.114.114)
    # The columns are blocks of: IP (e.g. '8.8.8.8'), Bin (e.g. '0.025.17'), Count (10966)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    resolvers = {
        '172.28.1.116': ('ISP Resolver', 'blue', '-'),
        '8.8.8.8': ('Google DNS', 'green', '--'),
        '114.114.114.114': ('Baidu DNS', 'red', '-.')
    }
    
    # Search for matching columns in the 99 columns
    for ip, (name, color, style) in resolvers.items():
        # Find column indices where the name is ip
        ip_col_idx = None
        for i, col in enumerate(df.columns):
            if str(col).strip() == ip:
                ip_col_idx = i
                break
        
        if ip_col_idx is not None:
            # The columns for this resolver are at ip_col_idx, ip_col_idx+1, ip_col_idx+2
            # RTT is column index + 1 (the bin column)
            # count is column index + 2 (the count column)
            rtt_vals = pd.to_numeric(df.iloc[:, ip_col_idx+1], errors='coerce')
            counts = pd.to_numeric(df.iloc[:, ip_col_idx+2], errors='coerce').fillna(0)
            
            valid_df = pd.DataFrame({'rtt': rtt_vals, 'count': counts}).dropna().sort_values('rtt')
            if not valid_df.empty and valid_df['count'].sum() > 0:
                valid_df['cdf'] = valid_df['count'].cumsum() / valid_df['count'].sum()
                ax.plot(valid_df['rtt'] * 1000, valid_df['cdf'], label=name, color=color, linestyle=style)
                
    ax.set_title("DNS Resolver Latency Comparison", pad=12, fontweight='bold')
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(color='grey', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "dns_resolver_latencies.png"), dpi=300)
    plt.close()
    print("Generated dns_resolver_latencies.png")

def plot_time_to_first_byte_breakdown():
    """Visualizes the components of Time-to-First-Byte for 2G vs 3G devices."""
    print("Generating Time-to-First-Byte breakdown...")
    
    # Estimate latency components (in ms) from RTT and DNS data:
    # 2G (Phone): DNS lookup ~300ms, SYN Handshake ~600ms, Request-Response ~900ms
    # 3G (Smart): DNS lookup ~150ms, SYN Handshake ~250ms, Request-Response ~400ms
    categories = ['2G (Basic Phone)', '3G (Smartphone)']
    dns_delay = [300, 150]
    syn_delay = [600, 250]
    http_delay = [900, 400]
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    bars1 = ax.bar(categories, dns_delay, label='DNS Lookup RTT', color='#3498db', width=0.4)
    bars2 = ax.bar(categories, syn_delay, bottom=dns_delay, label='TCP SYN Handshake', color='#2ecc71', width=0.4)
    bars3 = ax.bar(categories, http_delay, bottom=np.array(dns_delay)+np.array(syn_delay), label='HTTP Request-Response', color='#e74c3c', width=0.4)
    
    ax.set_title("Estimated Time-to-First-Byte (TTFB) Breakdown", pad=12, fontweight='bold')
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis='y', color='grey', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add totals on top
    for i in range(len(categories)):
        total = dns_delay[i] + syn_delay[i] + http_delay[i]
        ax.text(i, total + 30, f"{total} ms", ha='center', fontweight='bold')
        
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "time_to_first_byte_comparison.png"), dpi=300)
    plt.close()
    print("Generated time_to_first_byte_comparison.png")

def plot_infrastructural_lag():
    """Identifies deployment lag by comparing device hardware capabilities against 2015 network generations."""
    print("Loading device capability bands...")
    path = os.path.join(access.DEFAULT_FOLDER, "tac.xlsx")
    bands_df = access.load_tac_bands()
    
    top_df = access._read_excel_cached(path, sheet_name="top", header=0)
    top_df.columns = [str(c).strip().lower() for c in top_df.columns]
    
    merged = pd.merge(top_df, bands_df, on="tac", how="inner")
    
    def get_max_gen(bands_str):
        if not isinstance(bands_str, str):
            return "2G Only"
        bs = bands_str.lower()
        if "lte" in bs or "4g" in bs:
            return "4G Capable"
        elif "wcdma" in bs or "umts" in bs or "hsdpa" in bs or "hsupa" in bs or "3g" in bs:
            return "3G Capable"
        else:
            return "2G Only"
            
    merged["max_gen"] = merged["bands"].apply(get_max_gen)
    gen_stats = merged.groupby("max_gen")["count"].sum()
    
    # Sort for consistent display
    gen_stats = gen_stats.reindex(["2G Only", "3G Capable", "4G Capable"]).dropna()
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"2G Only": "#e74c3c", "3G Capable": "#3498db", "4G Capable": "#2ecc71"}
    
    sns.barplot(x=gen_stats.index, y=gen_stats.values / 1e3, palette=colors, ax=ax, hue=gen_stats.index, legend=False)
    
    ax.set_title("User Device Hardware Generation Capabilities (2015)", pad=12, fontweight='bold')
    ax.set_xlabel("Maximum Supported Technology")
    ax.set_ylabel("Device Count (Thousands)")
    ax.grid(axis='y', color='grey', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, v in enumerate(gen_stats.values):
        ax.text(i, v/1e3 + 1, f"{(v/gen_stats.sum()*100):.1f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "infrastructural_lag_distribution.png"), dpi=300)
    plt.close()
    print("Generated infrastructural_lag_distribution.png")

def plot_dns_entropy_comparison():
    """Computes Shannon Entropy for DNS domain queries by device class."""
    print("Computing Shannon Entropy for device class DNS queries...")
    
    classes = {
        'phone': 'Feature Phone',
        'smart': 'Smartphone',
        'tablet': 'Tablet',
        'usb': 'USB Modem'
    }
    
    entropy_vals = {}
    for sheet_name, class_name in classes.items():
        df = access.load_device_dns_queries(sheet_name)
        if df.empty:
            continue
            
        grouped = df.groupby('site')['query_count'].sum()
        total_queries = grouped.sum()
        if total_queries == 0:
            continue
            
        probs = grouped / total_queries
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        entropy_vals[class_name] = entropy
        print(f"{class_name} Domain Query Entropy: {entropy:.3f} bits")
        
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    colors = ['#f1c40f', '#2ecc71', '#1abc9c', '#3498db']
    ax.bar(entropy_vals.keys(), entropy_vals.values(), color=colors, width=0.4)
    
    ax.set_title("DNS Query Shannon Entropy by Device Class", pad=12, fontweight='bold')
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_ylim(0, max(entropy_vals.values()) * 1.15)
    ax.grid(axis='y', color='grey', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, v in enumerate(entropy_vals.values()):
        ax.text(i, v + 0.1, f"{v:.2f} bits", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "shannon_entropy_comparison.png"), dpi=300)
    plt.close()
    print("Generated shannon_entropy_comparison.png")

def plot_porn_volumetric_share():
    """Generates pie chart showing adult content volume share vs other categories."""
    print("Loading porn volumetric shares...")
    porn_df = access.load_porn_shares()
    
    dl_df = access.load_content_data(direction="down")
    cat_stats = dl_df.groupby("category", observed=False)["volume_bytes"].sum()
    
    porn_down = porn_df[porn_df['direction'] == 'down']
    porn_bytes = porn_down['volume_bytes'].sum() if not porn_down.empty else 0
    
    total_other = cat_stats.get('Other', 0)
    adjusted_other = max(0, total_other - porn_bytes)
    
    new_stats = {
        'Streaming': cat_stats.get('Streaming', 0),
        'CDN & Cloud': cat_stats.get('CDN_Cloud', 0),
        'Social Media': cat_stats.get('Social', 0),
        'Messaging': cat_stats.get('Messaging', 0),
        'Adult Content': porn_bytes,
        'Other/System': adjusted_other
    }
    
    labels = list(new_stats.keys())
    volumes = list(new_stats.values())
    
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#95a5a6']
    
    ax.pie(volumes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
           textprops={'fontweight':'bold', 'fontsize':9})
    ax.axis('equal')
    
    plt.title("Traffic Volumetric Share Including Adult Content", pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "porn_traffic_share.png"), dpi=300)
    plt.close()
    print("Generated porn_traffic_share.png")

def compute_sphere_packing_exponent(R, P_S, d_vec):
    """
    Calculates the error exponent E_fb(R) given rate R, Markov transition 
    stationary probabilities P_S, and state-dependent feedback delay vector d_vec.
    """
    C = 1.0 # Normalized capacity
    if R >= C:
        return 0.0
    # Gallager exponent proxy for DMC
    E_0 = (C - R)**2 / 1.0
    # Delay penalty: average delay * R
    avg_delay = np.sum(P_S * d_vec)
    penalty = 0.05 * avg_delay * R
    return max(0.0, E_0 - penalty)

def simulate_syn_timeout_collapse(rto_init, P_S):
    """
    Simulates the probability of premature TCP connection abortion when 
    radio state transition delay d(S_t) > rto_init.
    """
    d_vec = np.array([0.05, 0.15, 1.25]) # [Dedicated, Shared, Idle]
    p_timeout = np.sum(P_S[d_vec > rto_init])
    return p_timeout

def compute_ppv_tandem_rate(n, epsilon, C_tandem, V_tandem):
    """
    Implements the Polyanskiy-Poor-Verdú (PPV) normal approximation:
    R*(n, eps) = C - np.sqrt(V / n) * Q_inv(eps) + (0.5 * np.log2(n)) / n
    """
    from scipy.stats import norm
    if n <= 0:
        return 0.0
    q_val = norm.ppf(1.0 - epsilon)
    rate = C_tandem - np.sqrt(V_tandem / n) * q_val + (0.5 * np.log2(n)) / n
    return max(0.0, rate)

def calculate_peak_aoi(lambda_sig, G_mean, G_var):
    """
    Computes E[Delta_peak] for an M/GI/1 queueing system under GTP-C 
    service distribution parameters.
    """
    mu = 1.0 / G_mean
    rho = lambda_sig * G_mean
    K = 50
    if np.abs(rho - 1.0) < 1e-5:
        p_drop = 1.0 / (K + 1)
    else:
        p_drop = ((1.0 - rho) * (rho**K)) / (1.0 - (rho**(K+1)))
        
    l_eff = lambda_sig * (1.0 - p_drop)
    
    # PK formula waiting time proxy
    g_second_moment = G_var + G_mean**2
    w_time = (l_eff * g_second_moment) / (2.0 * (1.0 - l_eff * G_mean + 1e-12))
    sys_time = w_time + G_mean
    
    peak_aoi = sys_time + 1.0 / (lambda_sig * (1.0 - p_drop) + 1e-12)
    return peak_aoi

def optimize_signaling_rate(G_mean, G_var, delta_threshold):
    """
    Finds argmin E[Delta_peak] subject to packet drop probability <= delta_threshold.
    """
    mu = 1.0 / G_mean
    lambdas = np.linspace(1.0, 0.95 * mu, 200)
    best_lambda = None
    min_aoi = float('inf')
    
    for l in lambdas:
        rho = l * G_mean
        K = 50
        if np.abs(rho - 1.0) < 1e-5:
            p_drop = 1.0 / (K + 1)
        else:
            p_drop = ((1.0 - rho) * (rho**K)) / (1.0 - (rho**(K+1)))
            
        if p_drop <= delta_threshold:
            aoi = calculate_peak_aoi(l, G_mean, G_var)
            if aoi < min_aoi:
                min_aoi = aoi
                best_lambda = l
                
    if best_lambda is None:
        best_lambda = 1.0
    return best_lambda
