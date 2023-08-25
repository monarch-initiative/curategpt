"""Chat with a Google Drive."""
import logging
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from curate_gpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)


# Authenticate and build the service
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


@dataclass
class GoogleDriveWrapper(BaseWrapper):
    """
    A wrapper to provide a search facade over Wikipedia.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "google_drive"

    google_drive_id: str = "0ABiscjZCrxjUUk9PVA"
    google_folder_id: str = None

    default_doc_types = ["application/vnd.google-apps.document"]

    max_text_length = 3000
    text_overlap = 200

    service: Any = None

    search_limit_multiplier: ClassVar[int] = 1

    def __post_init__(self):
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
                with open("token.json", "w") as token:
                    token.write(creds.to_json())
        self.service = build("drive", "v3", credentials=creds)

    def external_search(self, text: str, expand: bool = False, limit=None, **kwargs) -> List[Dict]:
        if limit is None:
            limit = 10
        if expand:
            logger.warning("Google Drive does not support expansion")
        service = self.service
        query = f"fullText contains '{text}' and mimeType = 'application/vnd.google-apps.document'"
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                corpora="drive",
                driveId=self.google_drive_id,
                fields="nextPageToken, files(id, name, mimeType)",
                includeItemsFromAllDrives=True,  # Add this line
                supportsAllDrives=True,
                # Add this line too
                pageToken=None,
            )
            .execute()
        )
        logger.debug(response)
        files = list(response.get("files", []))
        files = [file for file in files if file.get("mimeType", None) in self.default_doc_types]
        logger.info(f"Search returned {len(files)} files")
        if len(files) >= limit:
            files = files[:limit]  # Trim the results if necessary
            logger.info(f"Truncating to top {limit} results")
        return self.objects_from_files(files)

    def objects_from_files(self, files: List[Dict]) -> List[Dict]:
        service = self.service
        objs = []
        for file in files:
            file_id = file["id"]
            request = service.files().export_media(fileId=file_id, mimeType="text/plain")
            response = request.execute()
            response_text = response.decode("utf-8")
            objs.append(
                {
                    "id": file_id,
                    "url": f"https://docs.google.com/document/d/{file_id}/",
                    "label": file.get("name", ""),
                    "text": response_text,
                    "mime_type": file.get("mimeType", None),
                }
            )
        return self.split_objects(objs)

    def split_objects(self, objects: List[Dict]) -> List[Dict]:
        """
        Split objects with text above a certain length into multiple objects.

        :param objects:
        :return:
        """
        new_objects = []
        for obj in objects:
            if len(obj["text"]) > self.max_text_length:
                obj_id = obj["id"]
                text = obj["text"]
                n = 0
                while text:
                    new_obj = obj.copy()
                    n += 1
                    new_obj["id"] = f"{obj_id}#{n}"
                    new_obj["text"] = text[: self.max_text_length + self.text_overlap]
                    new_objects.append(new_obj)
                    text = text[self.max_text_length :]
            else:
                new_objects.append(obj)
        return new_objects

    def objects_by_ids(self, object_ids: List[str]) -> List[Dict]:
        files = [{"id: object_id"} for object_id in object_ids]
        return self.objects_from_files(files)
