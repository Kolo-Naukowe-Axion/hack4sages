"""Model construction for the TensorFlow ADC baseline adaptation."""

from __future__ import annotations

import math

import tensorflow as tf

from .constants import DEFAULT_FILTERS
from .quantum import QuantumCircuitLayer


@tf.keras.utils.register_keras_serializable(package="adc_crossgen_tf")
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, factor: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.factor = float(factor)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.cast(inputs, tf.float32) * self.factor

    def get_config(self) -> dict[str, float]:
        config = super().get_config()
        config.update({"factor": self.factor})
        return config


def build_model(
    spectrum_length: int,
    aux_dim: int,
    target_dim: int,
    dropout: float,
    qnn_qubits: int,
    qnn_depth: int,
    filters: tuple[int, ...] = DEFAULT_FILTERS,
    quantum_device_name: str = "default.qubit",
) -> tf.keras.Model:
    spectra_input = tf.keras.Input(shape=(spectrum_length,), name="spectra")
    aux_input = tf.keras.Input(shape=(aux_dim,), name="aux")

    x = tf.keras.layers.Reshape((spectrum_length, 1), name="spectra_reshape")(spectra_input)
    for block_index, channels in enumerate(filters, start=1):
        x = tf.keras.layers.Conv1D(channels, 3, activation="relu", name=f"conv_{block_index}_a")(x)
        x = tf.keras.layers.Conv1D(channels, 3, activation="relu", name=f"conv_{block_index}_b")(x)
        x = tf.keras.layers.MaxPooling1D(name=f"pool_{block_index}")(x)

    x = tf.keras.layers.Flatten(name="spectra_flatten")(x)
    fused = tf.keras.layers.Concatenate(name="spectra_aux_concat")([x, aux_input])
    fused = tf.keras.layers.Dense(500, activation="relu", name="fusion_dense")(fused)
    fused_dropout = tf.keras.layers.Dropout(dropout, name="fusion_dropout")(fused)

    quantum_angles = tf.keras.layers.Dense(qnn_qubits, activation="tanh", name="quantum_projection")(fused)
    quantum_angles = ScaleLayer(math.pi, name="quantum_angle_scale")(quantum_angles)
    quantum_features = QuantumCircuitLayer(
        n_qubits=qnn_qubits,
        depth=qnn_depth,
        device_name=quantum_device_name,
        name="quantum_layer",
    )(quantum_angles)

    head = tf.keras.layers.Concatenate(name="classical_quantum_concat")([fused_dropout, quantum_features])
    head = tf.keras.layers.Dense(100, activation="relu", name="head_dense")(head)
    head = tf.keras.layers.Dropout(dropout, name="head_dropout")(head)
    outputs = tf.keras.layers.Dense(target_dim, activation=None, name="targets")(head)

    return tf.keras.Model(inputs=[spectra_input, aux_input], outputs=outputs, name="adc_crossgen_tf_quantum")
