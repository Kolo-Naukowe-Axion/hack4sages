"""Run hybrid quantum model training on GPU."""
from crossgen_hybrid_training import TrainingConfig, run_training_experiment

config = TrainingConfig(
    data_root="/workspace/ariel-ml-dataset",
    output_dir="/workspace/outputs",
    quantum_device="lightning.gpu",
    max_epochs=30,
)

result = run_training_experiment(config)

summary = result["summary"]
print("\n=== TRAINING COMPLETE ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

h = result["history_frame"]
print(f"\nLoss: {h['train_loss'].iloc[0]:.4f} -> {h['train_loss'].iloc[-1]:.4f}")
print(f"Val:  {h['inner_val_loss'].iloc[0]:.4f} -> {h['inner_val_loss'].iloc[-1]:.4f}")
print(f"\nArtifacts saved to: {summary['output_dir']}")
