import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict

log_dir = "."

class RequestDurationCDFAnalyzer:
    """Analyzer for request duration CDF generation."""
    
    def __init__(self):
        self.request_events = defaultdict(list)
        
    def parse_log_file(self, log_file_path):
        """Parse log file and organize events by request_id."""
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    # Only process events with request_id
                    if 'extra' in entry and 'request_id' in entry.get('extra', {}):
                        request_id = entry['extra']['request_id']
                        timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                        
                        event_data = {
                            'timestamp': timestamp,
                            'event': entry['event'],
                            'duration': entry.get('duration_sec', 0),
                            'extra': entry.get('extra', {}),
                            'workid': entry.get('workid'),
                            'step': entry.get('step')
                        }
                        
                        self.request_events[request_id].append(event_data)
                        
                except (json.JSONDecodeError, ValueError):
                    continue
    
    def plot_request_duration_cdf(self, figsize=(12, 8), save_path=None):
        """Create a CDF plot of request durations."""
        durations = []
        
        # Collect all request durations
        for request_id, events in self.request_events.items():
            complete_event = next((e for e in events if e['event'] == 'async_rollout_request_complete'), None)
            if complete_event:
                durations.append(complete_event['duration'])
        
        if not durations:
            print("No request duration data found")
            return None
        
        # Sort durations for CDF
        sorted_durations = np.sort(durations)
        n = len(sorted_durations)
        cdf_values = np.arange(1, n + 1) / n
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(sorted_durations, cdf_values, linewidth=2, color='steelblue', marker='o', markersize=3, alpha=0.7)
        
        ax.set_xlabel('Request Duration (seconds)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('CDF of Request Processing Durations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add percentile lines
        percentiles = [50, 90, 95, 99]
        colors = ['red', 'orange', 'purple', 'darkred']
        
        for p, color in zip(percentiles, colors):
            percentile_value = np.percentile(sorted_durations, p)
            ax.axvline(percentile_value, color=color, linestyle='--', alpha=0.7, 
                    label=f'P{p}: {percentile_value:.2f}s')
            ax.text(percentile_value, 0.1 + p/200, f'P{p}\n{percentile_value:.1f}s', 
                rotation=90, ha='right', va='bottom', fontsize=9)
        
        # Add statistics
        stats_text = f"""Statistics:
        Total Requests: {n}
        Min: {np.min(durations):.2f}s
        Max: {np.max(durations):.2f}s
        Mean: {np.mean(durations):.2f}s
        Median: {np.median(durations):.2f}s
        Std: {np.std(durations):.2f}s"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        ax.set_ylim(0, 1)
        
        # Add secondary y-axis with percentage labels
        ax2 = ax.twinx()
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Percentile (%)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CDF plot saved to {save_path}")
        
        return fig


def main():
    """Main function to run the CDF analysis."""
    analyzer = RequestDurationCDFAnalyzer()
    
    # Update with your actual log file path
    log_file = log_dir + "/step_32/worker_0.jsonl"
    
    print(f"Parsing log file: {log_file}")
    analyzer.parse_log_file(log_file)
    
    print(f"Found {len(analyzer.request_events)} requests with detailed events")
    
    print("\nGenerating request duration CDF...")
    analyzer.plot_request_duration_cdf(save_path="zpics/request_duration_cdf.png")
    plt.show()


if __name__ == "__main__":
    main()