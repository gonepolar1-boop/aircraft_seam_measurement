__all__ = ["run_gap_flush_pipeline", "preload_pipeline_model"]


def __getattr__(name: str):
    if name == "run_gap_flush_pipeline":
        from .gap_flush import run_gap_flush_pipeline

        return run_gap_flush_pipeline
    if name == "preload_pipeline_model":
        from .gap_flush import preload_pipeline_model

        return preload_pipeline_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
