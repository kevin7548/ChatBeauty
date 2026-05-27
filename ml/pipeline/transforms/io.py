"""Composite I/O transforms for reading/writing JSONL files."""

import json

import apache_beam as beam


class ReadJsonl(beam.PTransform):
    """Read a JSONL file and parse each line as JSON dict."""

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def expand(self, pcoll):
        return (
            pcoll
            | beam.io.ReadFromText(self.file_path)
            | beam.Map(json.loads)
        )


class WriteJsonl(beam.PTransform):
    """Serialize elements as JSON and write to a single JSONL file."""

    def __init__(self, file_path_prefix: str):
        super().__init__()
        self.file_path_prefix = file_path_prefix

    def expand(self, pcoll):
        return (
            pcoll
            | beam.Map(lambda x: json.dumps(x._asdict(), ensure_ascii=False))
            | beam.io.WriteToText(
                self.file_path_prefix,
                file_name_suffix=".jsonl",
                shard_name_template="",
            )
        )
