"""Run the whole pipeline end to end: preprocess -> train -> evaluate.

    python run_pipeline.py            # full data (~1.27M rows, ~5 min on a laptop)
    python run_pipeline.py --sample 200000   # quick run
"""
import argparse

from src import data_preprocessing, model_development, model_evaluation

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()
    data_preprocessing.main(args.sample)
    model_development.main()
    model_evaluation.main()
