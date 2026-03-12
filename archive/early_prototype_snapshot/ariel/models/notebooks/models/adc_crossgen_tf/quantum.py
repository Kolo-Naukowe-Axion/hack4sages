"""Quantum layer definitions for the TensorFlow ADC baseline adaptation."""

from __future__ import annotations

import tensorflow as tf


def import_pennylane():
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for the TensorFlow quantum baseline. Install models/requirements-tf-quantum.txt."
        ) from exc
    return qml


@tf.keras.utils.register_keras_serializable(package="adc_crossgen_tf")
class QuantumCircuitLayer(tf.keras.layers.Layer):
    """Batched expectation-value layer backed by a PennyLane QNode."""

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        device_name: str = "default.qubit",
        weight_scale: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_qubits = int(n_qubits)
        self.depth = int(depth)
        self.device_name = device_name
        self.weight_scale = float(weight_scale)
        self._qml = None
        self._device = None
        self._qnode = None

    def build(self, input_shape: tf.TensorShape) -> None:
        self.theta = self.add_weight(
            name="theta",
            shape=(self.depth, self.n_qubits, 3),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.weight_scale),
            trainable=True,
        )
        self._qml = import_pennylane()
        self._device = self._qml.device(self.device_name, wires=self.n_qubits)
        self._qnode = self._build_qnode()
        super().build(input_shape)

    def _build_qnode(self):
        qml = self._qml
        device = self._device
        n_qubits = self.n_qubits
        depth = self.depth

        @qml.qnode(device, interface="tf", diff_method="backprop")
        def circuit(inputs, weights):
            for wire in range(n_qubits):
                qml.RY(inputs[wire], wires=wire)
            for layer in range(depth):
                for wire in range(n_qubits):
                    qml.Rot(
                        weights[layer, wire, 0],
                        weights[layer, wire, 1],
                        weights[layer, wire, 2],
                        wires=wire,
                    )
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        return circuit

    def _run_single(self, sample: tf.Tensor) -> tf.Tensor:
        sample_outputs = self._qnode(sample, self.theta)
        if isinstance(sample_outputs, (list, tuple)):
            sample_outputs = tf.stack(sample_outputs, axis=-1)
        if sample_outputs.dtype.is_complex:
            sample_outputs = tf.math.real(sample_outputs)
        return tf.cast(sample_outputs, tf.float32)

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        if tf.is_symbolic_tensor(inputs):
            return tf.map_fn(
                self._run_single,
                inputs,
                fn_output_signature=tf.TensorSpec(shape=(self.n_qubits,), dtype=tf.float32),
            )
        batch_outputs = []
        for sample in tf.unstack(inputs, axis=0):
            batch_outputs.append(self._run_single(sample))
        return tf.stack(batch_outputs, axis=0)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape((input_shape[0], self.n_qubits))

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "n_qubits": self.n_qubits,
                "depth": self.depth,
                "device_name": self.device_name,
                "weight_scale": self.weight_scale,
            }
        )
        return config
