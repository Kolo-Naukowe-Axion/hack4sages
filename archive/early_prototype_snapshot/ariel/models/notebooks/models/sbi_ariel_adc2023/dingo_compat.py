"""Reuse Dingo compatibility helpers from the cross-generator FMPE workflow."""

from models.sbi_ariel_crossgen.dingo_compat import (  # noqa: F401
    build_posterior_model,
    build_resume_payload,
    expand_hidden_dims,
    load_posterior_model,
    move_model_to_device,
    prepare_model_settings,
)

