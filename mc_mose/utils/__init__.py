"""Shared utility functions for MC-MoSE."""

__all__ = [
    "build_run_paths",
    "collect_monthly_ic",
    "detach_to_cpu",
    "ensure_output_dirs",
    "export_test_predictions",
    "format_instrument",
    "json_default",
    "loss_parts_to_float",
    "masked_pairwise_rank_loss",
    "masked_pearson_corr",
    "move_to_device",
    "rank_1d",
    "save_checkpoint",
    "set_seed",
    "summarize_ic_lists",
    "summarize_monthly_values",
    "tensor_to_numpy",
    "write_summary_json",
]


def __getattr__(name):
    if name in {"detach_to_cpu", "move_to_device", "tensor_to_numpy"}:
        from . import batch_ops

        return getattr(batch_ops, name)
    if name in {
        "build_run_paths",
        "ensure_output_dirs",
        "export_test_predictions",
        "format_instrument",
        "json_default",
        "save_checkpoint",
        "write_summary_json",
    }:
        from . import artifact_io

        return getattr(artifact_io, name)
    if name in {"loss_parts_to_float", "masked_pairwise_rank_loss"}:
        from . import loss_utils

        return getattr(loss_utils, name)
    if name in {
        "collect_monthly_ic",
        "masked_pearson_corr",
        "rank_1d",
        "summarize_ic_lists",
        "summarize_monthly_values",
    }:
        from . import metric_utils

        return getattr(metric_utils, name)
    if name == "set_seed":
        from . import seed

        return getattr(seed, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
