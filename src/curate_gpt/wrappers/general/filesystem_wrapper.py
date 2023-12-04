"""Chat with a filesystem."""
import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterable, Iterator, Optional

from curate_gpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)


@dataclass
class FilesystemWrapper(BaseWrapper):

    """
    A wrapper over a filesystem.

    This is a static wrapper: it cannot be searched
    """

    name: ClassVar[str] = "filesystem"
    root_directory: str = None
    glob: str = None

    skip_unprocessable = True

    max_text_length = 3000
    text_overlap = 200

    search_limit_multiplier: ClassVar[int] = 1

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        path = self.root_directory or "."
        if self.glob:
            files = glob.glob(os.path.join(path, "**", self.glob), recursive=True)
        else:
            files = []
            for dirpath, _dirnames, filenames in os.walk(path):
                for filename in filenames:
                    files.append(os.path.join(dirpath, filename))
        import textract

        for file in set(files):
            try:
                # Extract text from the file
                ex = file.split(".")[-1]
                if ex in ["py", "md"]:
                    text_utf8 = open(file).read()
                else:
                    text = textract.process(file)
                    text_utf8 = text.decode("utf-8")
                path = Path(file)
                stat = path.lstat()
                obj = {
                    "id": file,
                    "name": path.name,
                    "text": text_utf8,
                    "parent": str(path.parent),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
                yield from self.split_objects([obj])
            except Exception as e:
                if self.skip_unprocessable:
                    logger.warning(f"Failed to extract text from {file}. Reason: {e}")
                else:
                    raise e
