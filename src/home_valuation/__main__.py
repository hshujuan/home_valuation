import argparse
from pathlib import Path

from .pipeline import run_study, save_study
from .simulation import Config


def main():
    parser = argparse.ArgumentParser(description="Synthetic home valuation and causal pricing lab")
    parser.add_argument("command", choices=["pipeline", "two-sided"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=None,
                        help="Override sizes: minimum 300 for pipeline, 2400 for two-sided")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--monte-carlo", type=int, default=None)
    parser.add_argument("--write-report", action="store_true",
                        help="Also update the public synthetic result tables and figures under docs")
    args = parser.parse_args()
    if args.command == "two-sided":
        from .two_sided import run_two_sided, save_two_sided
        from .two_sided_simulation import TwoSidedConfig
        if args.bootstrap is not None or args.monte_carlo is not None:
            parser.error("Resampling flags belong to the original pipeline, not the two-sided command.")
        if args.size is not None and args.size < 2400:
            parser.error("The linked extension needs at least 2400 prospects per cohort.")
        sizes = (dict.fromkeys(
            ["n_market", "n_development", "n_continuation", "n_evaluation"], args.size)
            if args.size is not None else {})
        output = args.output if args.output is not None else Path("outputs") / "two_sided"
        extension = run_two_sided(TwoSidedConfig(seed=args.seed, **sizes))
        save_two_sided(extension, output, args.write_report)
        print(f"Linked two-sided extension written to {output.resolve()}")
        print(extension.tables["policy_evaluation"].to_string(index=False))
        print("The original experiment is unchanged. No source documents were read or published.")
        return
    args.output = args.output if args.output is not None else Path("outputs") / "reference"
    args.bootstrap = args.bootstrap if args.bootstrap is not None else 30
    args.monte_carlo = args.monte_carlo if args.monte_carlo is not None else 30
    if args.bootstrap not in [0] and args.bootstrap < 10:
        parser.error("--bootstrap must be zero or at least 10")
    if args.monte_carlo < 0:
        parser.error("--monte-carlo cannot be negative")
    if args.size is not None and args.size < 300:
        parser.error("--size must be at least 300")
    sizes = dict.fromkeys(["n_market", "n_sellers", "n_resale", "n_iv"], args.size) if args.size else {}
    config = Config(seed=args.seed, **sizes)
    study = run_study(config, args.bootstrap, args.monte_carlo)
    save_study(study, args.output, args.write_report)
    print(f"Synthetic study written to {args.output.resolve()}")
    print(study.tables["causal_effects"][["design", "method", "estimate", "oracle_ate"]].to_string(index=False))
    print("No source documents were read and nothing was published to GitHub.")


if __name__ == "__main__":
    main()
