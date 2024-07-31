"""Chat with a KB."""

import json
import logging
import os
from dataclasses import dataclass, field
from time import sleep
from typing import ClassVar, Dict, Iterable, Iterator, Optional, List

import requests
import requests_cache
from oaklib import BasicOntologyInterface, get_adapter

from curate_gpt.wrappers import BaseWrapper

URL = "https://api.fairsharing.org"
SIGNIN_URL = f"{URL}/users/sign_in"
FETCH_URL = f"{URL}/fairsharing_records"


logger = logging.getLogger(__name__)


MAX_ID = 10000


@dataclass
class FAIRSharingWrapper(BaseWrapper):
    """
    A wrapper over the NMDC Biosample API.
    """

    name: ClassVar[str] = "fairsharing"

    default_object_type = "Metadata"

    session: requests_cache.CachedSession = field(
        default_factory=lambda: requests_cache.CachedSession("fairsharing")
    )

    user: str = field(default=os.getenv("FAIRSHARING_USER"))

    password: str = field(default=os.getenv("FAIRSHARING_PASSWORD"))

    _jwt: str = None

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        exclude = [462]
        for id in range(1, MAX_ID):
            if id in exclude:
                continue
            obj = self.object_by_id(id)
            if obj:
                yield obj

    @property
    def jwt(self) -> str:
        if self._jwt is None:
            payload = {"user": {"login": self.user, "password": self.password}}
            headers = {"Accept": "application/json", "Content-Type": "application/json"}

            response = requests.request(
                "POST", SIGNIN_URL, headers=headers, data=json.dumps(payload)
            )

            # Get the JWT from the response.text to use in the next part.
            data = response.json()
            self._jwt = data["jwt"]

        return self._jwt

    def object_by_id(self, object_id: str) -> Optional[Dict]:

        url = f"{FETCH_URL}/{object_id}"

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer {0}".format(self.jwt),
        }

        # sleep(0.05)
        response = self.session.request("GET", url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Could not download records for {object_id} from {url}")
            return None
