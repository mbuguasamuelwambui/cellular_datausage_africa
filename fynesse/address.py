import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import os
from . import access

def categorize_spatial_nodes(df):
    """Clusters physical base stations by traffic volume to identify spatial infrastructure stress."""
    # Group by base station device to get aggregate bytes and message counts
    node_stats = df.groupby('device')['val'].agg(['sum', 'count']).reset_index()
    
    # Fit KMeans (2 clusters) on log-scaled traffic volume
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(np.log1p(node_stats[['sum']]))
    
    # Map clusters to human-readable names (higher load center = Urban)
    centers = kmeans.cluster_centers_
    urban_label = 1 if centers[1][0] > centers[0][0] else 0
    node_stats['Category'] = [r'Urban/High-Load Node' if c == urban_label else r'Rural/Low-Load Node' for c in kmeans.labels_]
    
    # Academic bar or scatter plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=node_stats, x='device', y='sum', hue='Category', palette='Set1', alpha=0.6, ax=ax)
    ax.set_yscale('log')
    ax.set_title("Spatial Node Clustering: Infrastructure Load Distribution", fontweight='bold')
    ax.set_ylabel("Total Traffic Volume (Bytes, Log Scale)")
    ax.set_xlabel("Unique Base Station / Node ID")
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(access.OUTPUT_PLOTS_FOLDER, "spatial_nodes_clustering.png"), format="png", dpi=300)
    plt.close()
    
    return node_stats

def identify_researcher_windows(df):
    """Recommends the statistically optimal window for bulk academic data transfers."""
    useful_hourly = df[df['domain'] != 'Other/System'].groupby('hour')['val'].mean()
    best_hour = useful_hourly.rolling(window=4).mean().idxmin()
    
    print("\n--- STRATEGIC RECOMMENDATION ---")
    print(f"Optimal Network Utilization Window: {best_hour-4:02d}:00 to {best_hour:02d}:00")
    print("Benefit: Leverages off-peak capacity to avoid active congestion interference.")
    return best_hour

def classify_user_sessions(folder=access.DEFAULT_FOLDER) -> pd.DataFrame:
    """Classifies user connection sessions into short (signaling/chat) vs long (payload/streaming) sessions."""
    path = os.path.join(access.TXT_FOLDER, "tcpflow_stats_bw.txt")
    print("Classifying user sessions...")
    
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                time_idx = int(parts[0])
                device = int(parts[1])
                subparts = parts[2].split('\t')
                if len(subparts) == 2:
                    direction = subparts[0]
                    volume = float(subparts[1])
                    throughput = float(parts[3])
                    
                    # Classify: Short vs Long
                    category = "Short/Chat" if (volume < 100000 or throughput < 50000) else "Long/Streaming"
                    data.append({
                        'time': time_idx,
                        'device': device,
                        'direction': direction,
                        'volume': volume,
                        'throughput': throughput,
                        'category': category
                    })
                    
    df = pd.DataFrame(data)
    summary = df.groupby('category').agg(
        flow_count=('volume', 'count'),
        total_volume_gb=('volume', lambda x: x.sum() / 1e9),
        avg_throughput_kbps=('throughput', lambda x: x.mean() / 1000)
    )
    
    print("\n--- USER SESSION CLASSIFICATION ---")
    print(summary)
    return summary

def analyze_dns_resolver_efficiencies(folder=access.DEFAULT_FOLDER):
    """Analyzes query distribution and latencies across key DNS resolvers."""
    print("Analyzing DNS resolver query distributions...")
    df = access.load_resolver_rtt_histograms(folder)
    
    resolvers = ['172.28.1.116', '8.8.8.8', '114.114.114.114']
    resolver_names = {
        '172.28.1.116': 'ISP Local Resolver',
        '8.8.8.8': 'Google Public DNS',
        '114.114.114.114': 'Baidu DNS (China)'
    }
    
    print("\n--- DNS RESOLVER PERFORMANCE SUMMARY ---")
    for ip in resolvers:
        ip_col_idx = None
        for i, col in enumerate(df.columns):
            if str(col).strip() == ip:
                ip_col_idx = i
                break
                
        if ip_col_idx is not None:
            rtt_vals = pd.to_numeric(df.iloc[:, ip_col_idx+1], errors='coerce')
            counts = pd.to_numeric(df.iloc[:, ip_col_idx+2], errors='coerce').fillna(0)
            
            valid_df = pd.DataFrame({'rtt': rtt_vals, 'count': counts}).dropna().sort_values('rtt')
            total_queries = valid_df['count'].sum()
            
            if total_queries > 0:
                valid_df['cum_pct'] = valid_df['count'].cumsum() / total_queries
                median_rtt = valid_df[valid_df['cum_pct'] >= 0.50].iloc[0]['rtt'] * 1000
                print(f"{resolver_names[ip]} ({ip}): Total Queries={total_queries:,.0f} | Median Latency={median_rtt:.1f} ms")

def analyze_brand_geopolitics(folder=access.DEFAULT_FOLDER):
    """Analyzes the correlation between Chinese phone brands and default DNS query patterns."""
    print("Running brand geopolitics analysis...")
    
    # Load device census (contains Brand counts)
    brand_df = access.load_device_brands(folder)
    
    # Load DNS query listings for feature phone ('phone') and smartphone ('smart')
    phone_dns = access.load_device_dns_queries('phone', folder)
    smart_dns = access.load_device_dns_queries('smart', folder)
    
    print("\n--- BRAND DISTRIBUTION ---")
    print(brand_df.head(5))
    
    print("\n--- GEOPOLITICAL DOMAIN RESOLUTION SUMMARY ---")
    for name, df in [('Feature Phone (2G)', phone_dns), ('Smartphone (3G)', smart_dns)]:
        if df.empty:
            continue
            
        total_queries = df['query_count'].sum()
        
        # Segment by destination server ecosystem
        chinese_queries = df[df['site'].str.contains('baidu|duba|qq|taobao|alipay|ucweb|sina|weibo', case=False, na=False)]['query_count'].sum()
        western_queries = df[df['site'].str.contains('google|apple|icloud|yahoo|facebook|live|microsoft|twitter', case=False, na=False)]['query_count'].sum()
        
        print(f"\nDevice Class: {name} | Total Queries Analyzed: {total_queries:,.0f}")
        print(f"  Chinese Server Queries (Baidu/Duba/etc.): {chinese_queries:,.0f} ({(chinese_queries/total_queries*100):.3f}%)")
        print(f"  Western Server Queries (Google/Apple/etc.): {western_queries:,.0f} ({(western_queries/total_queries*100):.3f}%)")
