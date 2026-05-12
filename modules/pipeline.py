"""Modul untuk merangkai pipeline TFX."""

import os
from absl import logging
from tfx.orchestration import metadata, pipeline

PIPELINE_NAME = "rahmatdi-pipeline"

def init_local_pipeline(components, pipeline_root):
    """Init local pipeline"""
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        "----direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            os.path.join(pipeline_root, "metadata.sqlite")
        ),
        beam_pipeline_args=beam_args
    )
