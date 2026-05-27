"""
ChatBeauty Beam pipeline — main DAG orchestration.

Wires all transforms into the full pipeline:
  ReadSources → Parse → Validate → Aggregate → Join → Sink

Run locally:
    python -m ml.pipeline.run \
        --input-reviews=ml/data/raw/All_Beauty.jsonl \
        --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
        --input-keywords=ml/data/processed/keywords_train.jsonl \
        --output-dir=ml/data/processed/beam_output \
        --database-url=postgresql://chatbeauty:pass@localhost:5432/chatbeauty
"""

import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from ml.pipeline.options import ChatBeautyPipelineOptions
from ml.pipeline.transforms.io import ReadJsonl, WriteJsonl
from ml.pipeline.transforms.parse import (
    ParseKeywordFn,
    ParseMetadataFn,
    ParseReviewFn,
)
from ml.pipeline.transforms.validate import (
    ValidateKeywordFn,
    ValidateMetadataFn,
    ValidateReviewFn,
)
from ml.pipeline.transforms.aggregate import (
    AggregateKeywordsPerProduct,
    ComputeRerankingFeatures,
)
from ml.pipeline.transforms.join import JoinMetadataAndKeywords, MergeItemAndFeatures
from ml.pipeline.transforms.pairs import CreateTrainingPairs
from ml.pipeline.transforms.sink import WriteToPostgreSQL

logger = logging.getLogger(__name__)


def run(argv=None):
    options = PipelineOptions(argv)
    custom = options.view_as(ChatBeautyPipelineOptions)

    with beam.Pipeline(options=options) as p:
        # ── READ & PARSE & VALIDATE (3 parallel sources) ──

        reviews = (
            p
            | "ReadReviews" >> ReadJsonl(custom.input_reviews)
            | "ParseReviews" >> beam.ParDo(ParseReviewFn())
            | "ValidateReviews" >> beam.ParDo(ValidateReviewFn())
        )

        metadata = (
            p
            | "ReadMetadata" >> ReadJsonl(custom.input_metadata)
            | "ParseMetadata" >> beam.ParDo(ParseMetadataFn())
            | "ValidateMetadata" >> beam.ParDo(ValidateMetadataFn())
        )

        keywords = (
            p
            | "ReadKeywords" >> ReadJsonl(custom.input_keywords)
            | "ParseKeywords" >> beam.ParDo(ParseKeywordFn())
            | "ValidateKeywords" >> beam.ParDo(ValidateKeywordFn())
        )

        # ── AGGREGATE KEYWORDS PER PRODUCT (CombinePerKey) ──

        agg_keywords = keywords | "AggregateKeywords" >> AggregateKeywordsPerProduct()

        # ── JOIN METADATA + KEYWORDS → ITEMS (CoGroupByKey) ──

        items = (agg_keywords, metadata) | "JoinMetaKW" >> JoinMetadataAndKeywords()

        # ── COMPUTE RERANKING FEATURES (CombinePerKey + CoGroupByKey) ──

        features = (reviews, metadata) | "ComputeFeatures" >> ComputeRerankingFeatures()

        # ── MERGE ITEMS + FEATURES → PRODUCT RECORDS (CoGroupByKey) ──

        products = (items, features) | "MergeProducts" >> MergeItemAndFeatures()

        # ── SINK: WRITE TO POSTGRESQL ──

        if custom.database_url:
            products | "WriteToDB" >> beam.ParDo(
                WriteToPostgreSQL(custom.database_url)
            )
        else:
            logger.warning("No --database-url provided; skipping DB write.")

        # ── SINK: TRAINING PAIRS → JSONL ──

        pairs = (keywords, items) | "CreatePairs" >> CreateTrainingPairs(
            max_keywords=custom.max_keywords
        )
        pairs | "WritePairs" >> WriteJsonl(custom.output_dir + "/training_pairs")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
