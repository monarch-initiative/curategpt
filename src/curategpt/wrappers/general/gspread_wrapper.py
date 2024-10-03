"""Chat with a Google Drive."""

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterable, Iterator, Optional

import gspread
from curategpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)


# Authenticate and build the service
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


@dataclass
class GSpreadWrapper(BaseWrapper):
    """
    A wrapper to provide a search facade over gspread layer onto google sheets.

    Currently static, but could be made dynamic in the future.
    """

    name: ClassVar[str] = "gspread"

    sheet_name: str = None
    worksheet_name: str = None

    service: Any = None
    gc = None

    def __post_init__(self):
        if self.source_locator:
            if "/" in self.source_locator:
                self.sheet_name, self.worksheet_name = self.source_locator.split("/")
            else:
                self.sheet_name = self.source_locator
        self.gc = gspread.service_account()

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        logger.info(f"Loading {self.sheet_name}/{self.worksheet_name}")
        wks = self.gc.open(str(self.sheet_name)).worksheet(self.worksheet_name)
        yield from wks.get_all_records()
