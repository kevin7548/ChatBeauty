# ml/pipeline/

Apache Beam data pipeline (DirectRunner, offline). Entry point: `run.py`.

## DAG
```
3 parallel sources: ReadJsonl → Parse{Review,Metadata,Keyword}Fn → Validate*Fn
  keywords → AggregateKeywordsPerProduct ─┐
                                          ├─ JoinMetadataAndKeywords → items ─┐
  reviews+metadata → ComputeRerankingFeatures → features ────────────────────┤
                                            MergeItemAndFeatures → products
  products → WriteToPostgreSQL            (if --database-url)
  keywords+items → CreateTrainingPairs → WriteJsonl(training_pairs)
```
Order: **Parse → Validate → Aggregate → Join → Sink**. Two sinks: the `products` table and
`training_pairs.jsonl` (~1M BGE-M3 fine-tuning pairs).

## Files
`run.py` (wiring), `options.py` (`ChatBeautyPipelineOptions` CLI args), `schemas.py`
(Pydantic records), `transforms/` (`parse, validate, aggregate, join, pairs, io, sink`).

## Run
```bash
python -m ml.pipeline.run \
  --input-reviews=ml/data/raw/All_Beauty.jsonl \
  --input-metadata=ml/data/raw/meta_All_Beauty.jsonl \
  --input-keywords=ml/data/processed/keywords_train.jsonl \
  --output-dir=ml/data/processed/beam_output \
  --database-url=postgresql://chatbeauty:chatbeauty@localhost:5432/chatbeauty
```

## Docs
`docs/ml-pipeline.md`
