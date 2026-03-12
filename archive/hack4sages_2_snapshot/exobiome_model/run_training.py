"""CLI entrypoint for training the hybrid quantum-classical model."""

from crossgen_hybrid_training import TrainingConfig, run_training_experiment

if __name__ == "__main__":
    config = TrainingConfig(data_root="data")
    result = run_training_experiment(config)
    print(f"\nDone. Test RMSE mean: {result['summary']['test_rmse_mean']:.4f}")
    print(f"Artifacts saved to: {result['summary']['output_dir']}")
