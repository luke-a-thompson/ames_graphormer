from typing import Self

from tensorboardX import SummaryWriter


class LoggingConfig:
    def __init__(self, logdir: str):
        self.logdir = logdir
        self.flush_secs = None
        self.comment = None
        self.purge_step = None
        self.max_queue = None
        self.filename_suffix = None
        self.write_to_disk = None

    def with_flush_secs(self, flush_secs: int) -> Self:
        self.flush_secs = flush_secs
        return self

    def with_comment(self, comment: str) -> Self:
        self.comment = comment
        return self

    def with_purge_step(self, purge_step: int) -> Self:
        self.purge_step = purge_step
        return self

    def with_max_queue(self, max_queue: int) -> Self:
        self.max_queue = max_queue
        return self

    def with_filename_suffix(self, filename_suffix: str) -> Self:
        self.filename_suffix = filename_suffix
        return self

    def with_write_to_disk(self, write_to_disk: bool) -> Self:
        self.write_to_disk = write_to_disk
        return self

    def build(self) -> SummaryWriter:
        writer_params = {}

        if self.comment is not None:
            writer_params["comment"] = self.comment
        if self.flush_secs is not None:
            writer_params["flush_secs"] = self.flush_secs
        if self.purge_step is not None:
            writer_params["purge_step"] = self.purge_step
        if self.max_queue is not None:
            writer_params["max_queue"] = self.max_queue
        if self.filename_suffix is not None:
            writer_params["filename_suffix"] = self.filename_suffix
        if self.write_to_disk is not None:
            writer_params["write_to_disk"] = self.write_to_disk

        return SummaryWriter(self.logdir, **writer_params)
