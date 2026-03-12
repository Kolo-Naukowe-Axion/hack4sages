#!/bin/bash
set -e

echo "=== Installing dependencies ==="
pip install -q pennylane pennylane-lightning-gpu pandas h5py scikit-learn matplotlib pyarrow

echo "=== Extracting data ==="
cd /workspace
tar xzf quantum_upload.tar.gz

echo "=== Sanity check ==="
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')
import pennylane as qml
print(f'PennyLane: {qml.__version__}')
from crossgen_hybrid_training import TrainingConfig
c = TrainingConfig(data_root='/workspace/ariel-ml-dataset', quantum_device='lightning.gpu')
print(f'Qubits: {c.qnn_qubits}, Depth: {c.qnn_depth}, Batch: {c.train_batch_size}')
print('Sanity check PASSED')
"

echo "=== Ready to train ==="
echo "Run: cd /workspace && python run_training.py"
