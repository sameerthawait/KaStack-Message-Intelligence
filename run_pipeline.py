"""CLI entrypoint: python run_pipeline.py <messages.csv> [output_dir]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import run_pipeline

if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/messages.csv")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output")
    summary = run_pipeline(csv_path, output_dir)
    print("\n=== Pipeline Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
