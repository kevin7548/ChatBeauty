"""
Pipeline options for ChatBeauty Beam pipeline.

Subclasses PipelineOptions to add custom CLI arguments.
Paths work as local filesystem (DirectRunner) or gs:// (DataflowRunner).
"""

from apache_beam.options.pipeline_options import PipelineOptions


class ChatBeautyPipelineOptions(PipelineOptions):

    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            "--input-reviews",
            required=True,
            help="Path to All_Beauty.jsonl (raw reviews)",
        )
        parser.add_argument(
            "--input-metadata",
            required=True,
            help="Path to meta_All_Beauty.jsonl (product metadata)",
        )
        parser.add_argument(
            "--input-keywords",
            required=True,
            help="Path to keywords_train.jsonl (pre-computed LLM keywords)",
        )
        parser.add_argument(
            "--output-dir",
            required=True,
            help="Output directory for training_pairs.jsonl",
        )
        parser.add_argument(
            "--database-url",
            default=None,
            help="PostgreSQL connection URL (e.g. postgresql://user:pass@host:5432/db). "
            "If not provided, skips DB write.",
        )
        parser.add_argument(
            "--max-keywords",
            type=int,
            default=20,
            help="Max keywords per training query (default: 20)",
        )
