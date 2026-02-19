#!/usr/bin/env python3
"""
H2 Pilot Study Analysis Script

Analyzes annotations.csv and generates:
- Summary statistics
- LaTeX tables for the paper appendix
- Statistical tests (Fisher's exact)
- Matplotlib visualizations

Usage:
    python analyze_pilot.py [--output-dir OUTPUT_DIR]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

# Try to import scipy for statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Statistical tests will be skipped.")

# Try to import matplotlib for visualizations
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


def load_annotations(filepath: Path) -> list[dict]:
    """Load annotations from CSV file."""
    annotations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows
            if not row.get('paper_id'):
                continue
            # Parse numeric fields
            method_rung = row.get('method_rung')
            row['method_rung'] = int(method_rung) if method_rung and method_rung != 'NA' else None
            claim_rung = row.get('claim_rung')
            row['claim_rung'] = int(claim_rung) if claim_rung and claim_rung != 'NA' else None
            overclaim = row.get('gap_score')
            row['gap_score'] = int(overclaim) if overclaim and overclaim != 'NA' else None
            row['confidence'] = int(row['confidence']) if row.get('confidence') else None
            row['claim_prominence'] = int(row['claim_prominence']) if row.get('claim_prominence') else None
            # Parse replication status
            repl = row.get('replication_status', '').strip()
            if repl == 'NA' or repl == '':
                row['replication_status'] = None
            else:
                row['replication_status'] = float(repl)
            annotations.append(row)
    return annotations


def load_candidate_papers(filepath: Path) -> dict[str, dict]:
    """Load candidate papers metadata."""
    papers = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('paper_id'):
                papers[row['paper_id']] = row
    return papers


def compute_overclaim_distribution(annotations: list[dict]) -> dict:
    """Compute overclaim score distribution."""
    valid = [a for a in annotations if a['gap_score'] is not None]
    total = len(valid)

    distribution = defaultdict(int)
    for a in valid:
        distribution[a['gap_score']] += 1

    return {
        'total': total,
        'distribution': dict(distribution),
        'percentages': {k: v / total * 100 for k, v in distribution.items()}
    }


def compute_paper_level_stats(annotations: list[dict], candidate_papers: dict) -> list[dict]:
    """Compute paper-level statistics."""
    paper_claims = defaultdict(list)
    for a in annotations:
        paper_claims[a['paper_id']].append(a)

    paper_stats = []
    for paper_id, claims in paper_claims.items():
        valid_claims = [c for c in claims if c['gap_score'] is not None]
        overclaiming_claims = [c for c in valid_claims if c['gap_score'] > 0]

        # Get replication status (should be same for all claims in a paper)
        repl_statuses = [c['replication_status'] for c in claims if c['replication_status'] is not None]
        repl_status = repl_statuses[0] if repl_statuses else None

        # Get paper metadata
        meta = candidate_papers.get(paper_id, {})

        paper_stats.append({
            'paper_id': paper_id,
            'title': meta.get('title', 'Unknown'),
            'authors': meta.get('authors', 'Unknown'),
            'year': meta.get('year', ''),
            'venue': meta.get('venue', ''),
            'primary_method': meta.get('primary_method', ''),
            'total_claims': len(claims),
            'valid_claims': len(valid_claims),
            'overclaiming_claims': len(overclaiming_claims),
            'overclaim_rate': len(overclaiming_claims) / len(valid_claims) if valid_claims else 0,
            'replication_status': repl_status,
            'replication_evidence': claims[0].get('replication_evidence', '') if claims else ''
        })

    return sorted(paper_stats, key=lambda x: x['paper_id'])


def compute_overclaim_patterns(annotations: list[dict]) -> list[dict]:
    """Identify common overclaim patterns."""
    patterns = defaultdict(int)
    pattern_examples = defaultdict(list)

    for a in annotations:
        if a['gap_score'] and a['gap_score'] > 0:
            method = a.get('method_used', 'Unknown')
            # Extract key claim words
            claim_text = a.get('claim_text', '')

            # Identify pattern based on method and claim rung
            pattern_key = f"{method} (R{a['method_rung']}) -> R{a['claim_rung']}"
            patterns[pattern_key] += 1
            if len(pattern_examples[pattern_key]) < 2:
                pattern_examples[pattern_key].append(claim_text[:80] + '...' if len(claim_text) > 80 else claim_text)

    result = []
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        result.append({
            'pattern': pattern,
            'count': count,
            'examples': pattern_examples[pattern]
        })
    return result


def compute_location_stats(annotations: list[dict]) -> dict:
    """Compute overclaim statistics by claim location."""
    location_stats = defaultdict(lambda: {'total': 0, 'overclaim_sum': 0})

    for a in annotations:
        loc = a.get('claim_location', 'unknown')
        if a['gap_score'] is not None:
            location_stats[loc]['total'] += 1
            location_stats[loc]['overclaim_sum'] += a['gap_score']

    result = {}
    for loc, stats in location_stats.items():
        result[loc] = {
            'total': stats['total'],
            'mean_overclaim': stats['overclaim_sum'] / stats['total'] if stats['total'] > 0 else 0
        }
    return result


def compute_paper_type_stats(paper_stats: list[dict]) -> dict:
    """Compute overclaim statistics by paper type."""
    type_stats = defaultdict(lambda: {'papers': 0, 'overclaim_sum': 0})

    # Map methods to paper types
    method_to_type = {
        'Activation Patching': 'Circuit discovery',
        'Circuit Analysis': 'Circuit discovery',
        'Circuit Analysis + Ablation': 'Circuit discovery',
        'Causal Tracing': 'Knowledge localization',
        'Causal Tracing + ROME': 'Knowledge localization',
        'ROME Editing': 'Knowledge localization',
        'SAE Attribution': 'Evaluation/benchmark',
        'Interchange Intervention': 'Evaluation/benchmark',
        'Intervention Benchmark': 'Evaluation/benchmark',
        'Linear Probing': 'Applied/production',
        'Steering Vectors': 'Applied/production',
        'Steering + Probing': 'Applied/production',
        'Probing': 'Applied/production',
        'Weight Analysis': 'Other',
        'DAS/Interchange Intervention': 'Evaluation/benchmark',
        'Causal Mediation': 'Circuit discovery',
        'Causal Analysis': 'Circuit discovery',
    }

    for p in paper_stats:
        method = p.get('primary_method', 'Other')
        paper_type = method_to_type.get(method, 'Other')
        type_stats[paper_type]['papers'] += 1
        type_stats[paper_type]['overclaim_sum'] += p['overclaim_rate']

    result = {}
    for ptype, stats in type_stats.items():
        result[ptype] = {
            'papers': stats['papers'],
            'mean_overclaim': stats['overclaim_sum'] / stats['papers'] if stats['papers'] > 0 else 0
        }
    return result


def compute_replication_contingency(paper_stats: list[dict]) -> dict:
    """Compute contingency table for overclaim vs replication."""
    # Papers with replication evidence
    papers_with_repl = [p for p in paper_stats if p['replication_status'] is not None]

    # Define low/high overclaim threshold
    threshold = 0.25

    contingency = {
        'low_overclaim_replicated': 0,
        'low_overclaim_partial': 0,
        'low_overclaim_failed': 0,
        'high_overclaim_replicated': 0,
        'high_overclaim_partial': 0,
        'high_overclaim_failed': 0,
    }

    for p in papers_with_repl:
        overclaim_level = 'low' if p['overclaim_rate'] <= threshold else 'high'
        if p['replication_status'] == 0:
            repl_level = 'replicated'
        elif p['replication_status'] == 1:
            repl_level = 'failed'
        else:
            repl_level = 'partial'

        contingency[f'{overclaim_level}_overclaim_{repl_level}'] += 1

    # Compute totals
    contingency['low_overclaim_total'] = (
        contingency['low_overclaim_replicated'] +
        contingency['low_overclaim_partial'] +
        contingency['low_overclaim_failed']
    )
    contingency['high_overclaim_total'] = (
        contingency['high_overclaim_replicated'] +
        contingency['high_overclaim_partial'] +
        contingency['high_overclaim_failed']
    )

    # Fisher's exact test (replicated vs not replicated)
    if SCIPY_AVAILABLE:
        # 2x2 table: [low_repl, low_not_repl], [high_repl, high_not_repl]
        table = [
            [contingency['low_overclaim_replicated'],
             contingency['low_overclaim_partial'] + contingency['low_overclaim_failed']],
            [contingency['high_overclaim_replicated'],
             contingency['high_overclaim_partial'] + contingency['high_overclaim_failed']]
        ]
        if sum(table[0]) > 0 and sum(table[1]) > 0:
            odds_ratio, p_value = stats.fisher_exact(table)
            contingency['fisher_odds_ratio'] = odds_ratio
            contingency['fisher_p_value'] = p_value
        else:
            contingency['fisher_odds_ratio'] = None
            contingency['fisher_p_value'] = None

    return contingency


def generate_latex_overclaim_distribution(dist: dict) -> str:
    """Generate LaTeX table for overclaim distribution."""
    patterns = {
        0: 'Claim matches method',
        1: r'R2$\to$R3 typical',
        2: r'R1$\to$R3 (probing$\to$encodes)'
    }

    rows = []
    for score in sorted(dist['distribution'].keys()):
        count = dist['distribution'][score]
        pct = dist['percentages'][score]
        pattern = patterns.get(score, 'Other')
        label = {0: 'no overclaim', 1: 'mild overclaim', 2: 'strong overclaim'}.get(score, f'+{score}')
        rows.append(f"{score} ({label}) & {count} & {pct:.1f}\\% & {pattern} \\\\")

    return f"""\\begin{{table}}[h]
\\centering
\\caption{{Overclaim Score Distribution (N={dist['total']} claims)}}
\\label{{tab:overclaim-distribution}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Overclaim Score}} & \\textbf{{Count}} & \\textbf{{\\%}} & \\textbf{{Pattern}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def generate_latex_paper_type_table(type_stats: dict) -> str:
    """Generate LaTeX table for paper type statistics."""
    rows = []
    for ptype in ['Circuit discovery', 'Knowledge localization', 'Evaluation/benchmark', 'Applied/production', 'Other']:
        if ptype in type_stats:
            stats = type_stats[ptype]
            rows.append(f"{ptype} & {stats['papers']} & {stats['mean_overclaim']:.2f} \\\\")

    return f"""\\begin{{table}}[h]
\\centering
\\caption{{Mean Overclaim Rate by Paper Type}}
\\label{{tab:overclaim-by-type}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Paper Type}} & \\textbf{{Papers}} & \\textbf{{Mean Overclaim}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def generate_latex_contingency_table(contingency: dict) -> str:
    """Generate LaTeX contingency table."""
    fisher_note = ""
    if contingency.get('fisher_p_value') is not None:
        odds = contingency['fisher_odds_ratio']
        pval = contingency['fisher_p_value']
        fisher_note = f"Fisher's exact test: OR={odds:.2f}, p={pval:.3f}"

    # Extract values for readability
    low_repl = contingency['low_overclaim_replicated']
    low_part = contingency['low_overclaim_partial']
    low_fail = contingency['low_overclaim_failed']
    low_tot = contingency['low_overclaim_total']
    high_repl = contingency['high_overclaim_replicated']
    high_part = contingency['high_overclaim_partial']
    high_fail = contingency['high_overclaim_failed']
    high_tot = contingency['high_overclaim_total']

    return f"""\\begin{{table}}[h]
\\centering
\\caption{{Overclaim vs Replication Contingency Table}}
\\label{{tab:overclaim-replication}}
\\begin{{tabular}}{{lcccc}}
\\toprule
 & \\textbf{{Replicated}} & \\textbf{{Partial}} & \\textbf{{Failed}} & \\textbf{{Total}} \\\\
\\midrule
Low overclaim ($\\leq$25\\%) & {low_repl} & {low_part} & {low_fail} & {low_tot} \\\\
High overclaim ($>$25\\%) & {high_repl} & {high_part} & {high_fail} & {high_tot} \\\\
\\bottomrule
\\end{{tabular}}
\\\\[0.5em]
\\small {fisher_note}
\\end{{table}}"""


def generate_latex_paper_list(paper_stats: list[dict]) -> str:
    """Generate LaTeX table listing all papers."""
    rows = []
    for p in paper_stats:
        # Escape special characters in title
        title = p['title'][:50] + '...' if len(p['title']) > 50 else p['title']
        title = title.replace('&', '\\&').replace('_', '\\_').replace('%', '\\%')

        # Format replication status
        repl = p['replication_status']
        if repl is None:
            repl_str = 'N/A'
        elif repl == 0:
            repl_str = 'Full'
        elif repl == 1:
            repl_str = 'Failed'
        else:
            repl_str = 'Partial'

        rows.append(
            f"\\citet{{{p['paper_id'].replace('.', '')}}} & "
            f"{p['valid_claims']} & "
            f"{p['overclaim_rate']*100:.0f}\\% & "
            f"{repl_str} \\\\"
        )

    return f"""\\begin{{table}}[h]
\\centering
\\caption{{Papers Annotated in Pilot Study (N={len(paper_stats)})}}
\\label{{tab:pilot-study-papers}}
\\small
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Paper}} & \\textbf{{Claims}} & \\textbf{{Overclaim Rate}} & \\textbf{{Replication}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def generate_latex_location_table(location_stats: dict) -> str:
    """Generate LaTeX table for claim location statistics."""
    rows = []
    for loc in ['abstract', 'introduction', 'body', 'results', 'discussion', 'conclusion']:
        if loc in location_stats:
            stats = location_stats[loc]
            rows.append(f"{loc.capitalize()} & {stats['total']} & {stats['mean_overclaim']:.2f} \\\\")

    return f"""\\begin{{table}}[h]
\\centering
\\caption{{Mean Overclaim Score by Claim Location}}
\\label{{tab:overclaim-by-location}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Location}} & \\textbf{{Claims}} & \\textbf{{Mean Overclaim}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def compute_method_specific_stats(annotations: list[dict]) -> dict:
    """Compute overclaim statistics by method type."""
    method_stats = defaultdict(lambda: {'total': 0, 'overclaim_sum': 0, 'claims': []})

    for a in annotations:
        method = a.get('method_used', 'Unknown')
        if a['gap_score'] is not None:
            method_stats[method]['total'] += 1
            method_stats[method]['overclaim_sum'] += a['gap_score']
            method_stats[method]['claims'].append(a['gap_score'])

    result = {}
    for method, mstats in method_stats.items():
        if mstats['total'] >= 2:  # Only include methods with enough samples
            result[method] = {
                'total': mstats['total'],
                'mean_overclaim': mstats['overclaim_sum'] / mstats['total'],
                'overclaim_rate': sum(1 for c in mstats['claims'] if c > 0) / mstats['total']
            }
    return result


def compute_temporal_stats(paper_stats: list[dict]) -> dict:
    """Compute overclaim statistics by publication year."""
    year_stats = defaultdict(lambda: {'papers': 0, 'overclaim_sum': 0})

    for p in paper_stats:
        year = p.get('year', '')
        if year and p['valid_claims'] > 0:
            year_stats[year]['papers'] += 1
            year_stats[year]['overclaim_sum'] += p['overclaim_rate']

    result = {}
    for year, ystats in sorted(year_stats.items()):
        result[year] = {
            'papers': ystats['papers'],
            'mean_overclaim': ystats['overclaim_sum'] / ystats['papers'] if ystats['papers'] > 0 else 0
        }
    return result


def plot_overclaim_distribution(overclaim_dist: dict, output_path: Path) -> None:
    """Generate bar chart of overclaim score distribution."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot_overclaim_distribution: matplotlib not available")
        return

    scores = sorted(overclaim_dist['distribution'].keys())
    counts = [overclaim_dist['distribution'][s] for s in scores]
    percentages = [overclaim_dist['percentages'][s] for s in scores]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    bars = ax.bar(scores, counts, color=[colors[min(s, 2)] for s in scores], edgecolor='black')

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Overclaim Score (claim_rung - method_rung)', fontsize=12)
    ax.set_ylabel('Number of Claims', fontsize=12)
    ax.set_title(f'Overclaim Score Distribution (N={overclaim_dist["total"]} claims)', fontsize=14)
    ax.set_xticks(scores)
    ax.set_xticklabels(['0\n(matched)', '1\n(mild)', '2\n(strong)'][:len(scores)])

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='No overclaim'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='Mild overclaim (R2→R3)'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Strong overclaim (R1→R3)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")


def plot_method_confusion_matrix(annotations: list[dict], output_path: Path) -> None:
    """Generate heatmap of method_rung vs claim_rung."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot_method_confusion_matrix: matplotlib not available")
        return

    # Build confusion matrix
    matrix = np.zeros((3, 3), dtype=int)  # 3x3 for rungs 1, 2, 3
    for a in annotations:
        if a['method_rung'] is not None and a['claim_rung'] is not None:
            m_idx = a['method_rung'] - 1
            c_idx = a['claim_rung'] - 1
            if 0 <= m_idx < 3 and 0 <= c_idx < 3:
                matrix[m_idx, c_idx] += 1

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Number of Claims', rotation=-90, va="bottom")

    # Set ticks and labels
    rungs = ['R1\n(Observational)', 'R2\n(Interventional)', 'R3\n(Counterfactual)']
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(rungs)
    ax.set_yticklabels(rungs)

    ax.set_xlabel('Claim Rung', fontsize=12)
    ax.set_ylabel('Method Rung', fontsize=12)
    ax.set_title('Method Rung vs Claim Rung Confusion Matrix', fontsize=14)

    # Add text annotations
    for i in range(3):
        for j in range(3):
            color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
            ax.text(j, i, matrix[i, j], ha="center", va="center", color=color, fontsize=14)
            # Highlight overclaiming cells (above diagonal)
            if j > i:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor='red', linewidth=2, linestyle='--'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")


def plot_overclaim_by_year(temporal_stats: dict, output_path: Path) -> None:
    """Generate line plot of overclaim rate by publication year."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot_overclaim_by_year: matplotlib not available")
        return

    years = list(temporal_stats.keys())
    overclaim_rates = [temporal_stats[y]['mean_overclaim'] for y in years]
    paper_counts = [temporal_stats[y]['papers'] for y in years]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot overclaim rate as line
    color1 = '#3498db'
    ax1.plot(years, overclaim_rates, 'o-', color=color1, linewidth=2, markersize=8,
             label='Mean Overclaim Rate')
    ax1.set_xlabel('Publication Year', fontsize=12)
    ax1.set_ylabel('Mean Overclaim Rate', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1)

    # Add paper count as bars on secondary axis
    ax2 = ax1.twinx()
    color2 = '#95a5a6'
    ax2.bar(years, paper_counts, alpha=0.3, color=color2, label='Paper Count')
    ax2.set_ylabel('Number of Papers', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Overclaim Rate by Publication Year', fontsize=14)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")


def plot_replication_vs_overclaim(paper_stats: list[dict], output_path: Path) -> None:
    """Generate scatter plot of replication status vs overclaim rate."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot_replication_vs_overclaim: matplotlib not available")
        return

    # Filter papers with replication data
    papers_with_repl = [p for p in paper_stats if p['replication_status'] is not None]

    if len(papers_with_repl) < 3:
        print(f"Skipping plot_replication_vs_overclaim: only {len(papers_with_repl)} papers")
        return

    overclaim_rates = [p['overclaim_rate'] for p in papers_with_repl]
    repl_statuses = [p['replication_status'] for p in papers_with_repl]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by replication status
    colors = ['#2ecc71' if r == 0 else '#f39c12' if r == 0.5 else '#e74c3c' for r in repl_statuses]

    ax.scatter(overclaim_rates, repl_statuses, c=colors, s=100, edgecolors='black', alpha=0.7)

    # Add jitter for overlapping points
    for i, p in enumerate(papers_with_repl):
        ax.annotate(p['paper_id'][:10], (overclaim_rates[i], repl_statuses[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    ax.set_xlabel('Paper Overclaim Rate', fontsize=12)
    ax.set_ylabel('Replication Status (0=Full, 0.5=Partial, 1=Failed)', fontsize=12)
    ax.set_title(f'Replication Status vs Overclaim Rate (N={len(papers_with_repl)} papers)', fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['Replicated', 'Partial', 'Failed'])

    # Add trend line if scipy available
    if SCIPY_AVAILABLE and len(papers_with_repl) >= 5:
        slope, intercept, r_value, p_value, _ = stats.linregress(overclaim_rates, repl_statuses)
        x_line = np.array([0, 1])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', alpha=0.5, label=f'Trend (r={r_value:.2f}, p={p_value:.3f})')

    # Add legend for colors
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Replicated'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='Partial'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Failed')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")


def generate_latex_method_stats_table(method_stats: dict) -> str:
    """Generate LaTeX table for method-specific statistics."""
    rows = []
    for method in sorted(method_stats.keys(), key=lambda m: -method_stats[m]['total']):
        mstats = method_stats[method]
        if mstats['total'] >= 3:  # Only show methods with enough samples
            method_escaped = method.replace('_', '\\_').replace('&', '\\&')
            rows.append(
                f"{method_escaped} & {mstats['total']} & "
                f"{mstats['mean_overclaim']:.2f} & {mstats['overclaim_rate']*100:.0f}\\% \\\\"
            )

    return f"""\\begin{{table}}[h]
\\centering
\\caption{{Overclaim Statistics by Method Type}}
\\label{{tab:overclaim-by-method}}
\\small
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{Claims}} & \\textbf{{Mean Overclaim}} & \\textbf{{Overclaim Rate}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def print_summary(annotations: list[dict], paper_stats: list[dict],
                  overclaim_dist: dict, contingency: dict) -> None:
    """Print summary statistics to console."""
    print("=" * 60)
    print("H2 PILOT STUDY ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nTotal papers: {len(paper_stats)}")
    print(f"Total claims: {len(annotations)}")
    print(f"Valid claims (with overclaim score): {overclaim_dist['total']}")

    print("\n--- Overclaim Distribution ---")
    for score, count in sorted(overclaim_dist['distribution'].items()):
        pct = overclaim_dist['percentages'][score]
        print(f"  Score {score}: {count} ({pct:.1f}%)")

    overclaiming = sum(c for s, c in overclaim_dist['distribution'].items() if s > 0)
    print(f"\nTotal overclaiming: {overclaiming} ({overclaiming/overclaim_dist['total']*100:.1f}%)")

    print("\n--- Replication vs Overclaim ---")
    papers_with_repl = [p for p in paper_stats if p['replication_status'] is not None]
    print(f"Papers with replication evidence: {len(papers_with_repl)}")
    print(f"  Low overclaim, replicated: {contingency['low_overclaim_replicated']}")
    print(f"  Low overclaim, partial: {contingency['low_overclaim_partial']}")
    print(f"  High overclaim, replicated: {contingency['high_overclaim_replicated']}")
    print(f"  High overclaim, partial: {contingency['high_overclaim_partial']}")

    if contingency.get('fisher_p_value') is not None:
        print("\nFisher's exact test:")
        print(f"  Odds ratio: {contingency['fisher_odds_ratio']:.2f}")
        print(f"  p-value: {contingency['fisher_p_value']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze H2 pilot study annotations')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'output',
                        help='Directory for output files')
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    annotations_path = script_dir / 'annotations.csv'
    candidates_path = script_dir / 'candidate_papers.csv'

    # Load data
    print(f"Loading annotations from {annotations_path}")
    annotations = load_annotations(annotations_path)

    print(f"Loading candidate papers from {candidates_path}")
    candidate_papers = load_candidate_papers(candidates_path)

    # Compute statistics
    overclaim_dist = compute_overclaim_distribution(annotations)
    paper_stats = compute_paper_level_stats(annotations, candidate_papers)
    location_stats = compute_location_stats(annotations)
    type_stats = compute_paper_type_stats(paper_stats)
    contingency = compute_replication_contingency(paper_stats)
    method_stats = compute_method_specific_stats(annotations)
    temporal_stats = compute_temporal_stats(paper_stats)
    _ = compute_overclaim_patterns(annotations)  # Computed but not yet used in tables

    # Print summary
    print_summary(annotations, paper_stats, overclaim_dist, contingency)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    if MATPLOTLIB_AVAILABLE:
        print("\n--- Generating Visualizations ---")
        plot_overclaim_distribution(overclaim_dist, args.output_dir / 'overclaim_distribution.pdf')
        plot_method_confusion_matrix(annotations, args.output_dir / 'method_claim_matrix.pdf')
        plot_overclaim_by_year(temporal_stats, args.output_dir / 'overclaim_by_year.pdf')
        plot_replication_vs_overclaim(paper_stats, args.output_dir / 'replication_vs_overclaim.pdf')

    # Generate and save LaTeX tables
    tables = {
        'overclaim_distribution.tex': generate_latex_overclaim_distribution(overclaim_dist),
        'paper_type_stats.tex': generate_latex_paper_type_table(type_stats),
        'contingency_table.tex': generate_latex_contingency_table(contingency),
        'paper_list.tex': generate_latex_paper_list(paper_stats),
        'location_stats.tex': generate_latex_location_table(location_stats),
        'method_stats.tex': generate_latex_method_stats_table(method_stats),
    }

    for filename, content in tables.items():
        output_path = args.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Generated: {output_path}")

    # Save paper stats as CSV for reference
    csv_path = args.output_dir / 'paper_stats.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['paper_id', 'title', 'authors', 'year', 'venue', 'primary_method',
                      'valid_claims', 'overclaim_rate', 'replication_status']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(paper_stats)
    print(f"Generated: {csv_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
