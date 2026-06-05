"""
Intraday Liquidity Regimes Research - Master Run Script
=========================================================
Orchestrates the complete research pipeline.

Usage:
    python run_liquidity_regime_research.py --phase all
    python run_liquidity_regime_research.py --phase 1  # Data construction only
    python run_liquidity_regime_research.py --phase 2  # Regime identification only
"""

import argparse
import os
import sys

def run_phase1():
    """Phase 1: Data Construction"""
    print("\n" + "="*70)
    print("PHASE 1: DATA CONSTRUCTION")
    print("="*70)
    
    import construct_intraday_profiles
    
    # Build feature matrix
    feature_df = construct_intraday_profiles.build_feature_matrix(
        years=list(range(2015, 2026))
    )
    
    # Save
    output_path = os.path.join(construct_intraday_profiles.OUTPUT_DIR, "intraday_features.csv")
    feature_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    return feature_df


def run_phase2():
    """Phase 2: Regime Identification"""
    print("\n" + "="*70)
    print("PHASE 2: REGIME IDENTIFICATION")
    print("="*70)
    
    import identify_liquidity_regimes
    
    df = identify_liquidity_regimes.load_features()
    output_df, interp_df, eval_df = identify_liquidity_regimes.run_clustering_pipeline(
        df, validate=True
    )
    identify_liquidity_regimes.save_results(output_df, interp_df, eval_df)
    
    return output_df


def run_phase2b():
    """Phase 2b: Visualizations"""
    print("\n" + "="*70)
    print("PHASE 2b: REGIME VISUALIZATIONS")
    print("="*70)
    
    import visualize_regimes
    visualize_regimes.generate_all_visualizations()


def run_phase3():
    """Phase 3: Cross-Sectional Analysis"""
    print("\n" + "="*70)
    print("PHASE 3: CROSS-SECTIONAL ANALYSIS")
    print("="*70)
    
    import analyze_regime_determinants
    analyze_regime_determinants.run_determinants_analysis()


def run_phase4():
    """Phase 4: Evolution & Structural Change"""
    print("\n" + "="*70)
    print("PHASE 4: EVOLUTION & STRUCTURAL CHANGE")
    print("="*70)
    
    import track_regime_evolution
    track_regime_evolution.run_evolution_analysis()


def run_phase5():
    """Phase 5: Stress Analysis"""
    print("\n" + "="*70)
    print("PHASE 5: STRESS ANALYSIS")
    print("="*70)
    
    import analyze_stress_days
    analyze_stress_days.run_stress_analysis()


def run_phase6():
    """Phase 6: Robustness Checks"""
    print("\n" + "="*70)
    print("PHASE 6: ROBUSTNESS CHECKS")
    print("="*70)
    
    import robustness_checks
    robustness_checks.run_robustness_checks()


def run_all():
    """Run complete research pipeline."""
    print("\n" + "#"*70)
    print("# INTRADAY LIQUIDITY REGIMES RESEARCH")
    print("# NIFTY 500 (2015-2025)")
    print("#"*70)
    
    run_phase1()
    run_phase2()
    run_phase2b()
    run_phase3()
    run_phase4()
    run_phase5()
    run_phase6()
    
    print("\n" + "="*70)
    print("RESEARCH PIPELINE COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print("  - results/intraday_features.csv       (Feature matrix)")
    print("  - results/regime_assignments.csv      (Cluster labels)")
    print("  - results/regime_evolution.csv        (Yearly distribution)")
    print("  - results/transition_matrix.csv       (Regime transitions)")
    print("  - plots/regime_*.png                  (Visualizations)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run liquidity regime research')
    parser.add_argument('--phase', type=str, default='all',
                        help='Phase to run: 1, 2, 2b, 3, 4, 5, 6, or all')
    args = parser.parse_args()
    
    phase_map = {
        '1': run_phase1,
        '2': run_phase2,
        '2b': run_phase2b,
        '3': run_phase3,
        '4': run_phase4,
        '5': run_phase5,
        '6': run_phase6,
        'all': run_all
    }
    
    if args.phase not in phase_map:
        print(f"Invalid phase: {args.phase}")
        print(f"Valid phases: {list(phase_map.keys())}")
        sys.exit(1)
    
    phase_map[args.phase]()
