"""Chat with issues from a GitHub repository."""

import logging
import os
from dataclasses import dataclass
from time import sleep
from typing import ClassVar, Dict, Iterable, Iterator, List, Optional

import requests
import requests_cache
from curategpt.wrappers.base_wrapper import BaseWrapper
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class Comment(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    user: str = None
    body: str = None


class PullRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    number: int = None
    title: str = None
    user: str = None
    labels: List[str] = None
    state: str = None
    assignees: List[str] = None
    created_at: str = None
    body: str = None
    comments: List[Comment] = None


class Issue(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    number: int = None
    type: str = None
    title: str = None
    user: str = None
    labels: List[str] = None
    state: str = None
    assignees: List[str] = None
    created_at: str = None
    body: str = None
    # pull_request: str = None
    comments: List[Comment] = None


def pr_comments(self, pr_number: str) -> Iterator[Dict]:
    session = self.session
    url = f"https://api.github.com/repos/{self.owner}/{self.repo}/pulls/{pr_number}/comments"
    params = {"per_page": 100}

    while url:
        response = session.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        yield from response.json()
        url = response.links.get("next", {}).get("url")


def get_token(token: str = None) -> Optional[str]:
    if token:
        return token
    """Get token from env var"""
    token = os.environ.get("CURATEGPT_GITHUB_TOKEN")
    # if not token:
    #    raise ValueError("CURATEGPT_GITHUB_TOKEN env var not set")
    return token


@dataclass
class GitHubWrapper(BaseWrapper):
    """
    A wrapper to provide a search facade over GitHub.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "github"

    cache_name: ClassVar[str] = "github_requests"

    default_object_type = "Issue"

    session: requests.Session = None

    owner: str = None
    repo: str = None

    _repo_description: str = None

    def __post_init__(self):
        self.session = requests_cache.CachedSession(self.cache_name)
        if self.repo and "/" in self.repo:
            if self.owner:
                raise ValueError("Cannot specify both owner and a slash in repo")
            self.owner, self.repo = self.repo.split("/")

    @property
    def headers(self):
        token = get_token()
        hdr = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CurateGPT/0.0.1",
        }
        if not token:
            del hdr["Authorization"]
        logger.info(f"Header: {hdr}")
        return hdr

    @property
    def repo_description(self) -> str:
        if not self._repo_description:
            url = f"https://api.github.com/repos/{self.owner}/{self.repo}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            repo_data = response.json()
            self._repo_description = repo_data.get("description")
        return self._repo_description

    def external_search(
            self, text: str, expand: bool = True, limit=None, token: str = None, **kwargs
    ) -> List[Dict]:
        token = get_token(token)
        if limit is None:
            limit = 10
        if expand:
            logger.info(f"Expanding search term: {text} to create query")
            model = self.extractor.model
            q = (
                "I will give you a piece of text."
                "You will generate a semi-colon separated list of the 3 most relevant terms"
                f" to search the {self.owner}/{self.repo} repo on GitHub."
                "Keep this list minimal and precise. Use semi-colons to separate terms."
                "Use terms that within the domain of the repo."
                f"The description of the repo is: {self.repo_description}.\n"
                f"\n---\nHere is the text:\n{text}"
            )
            response = model.prompt(
                q,
                system="You are an agent to expand query terms to search github. ALWAYS SEPARATE WITH SEMI-COLONS",
            )
            terms = [f'"{x.strip()}"' for x in response.text().split(";")]
            search_term = " OR ".join(terms[:3])
        else:
            search_term = text
        logger.info(f"Constructed search term: {search_term}")
        url = f"https://api.github.com/search/issues?q={search_term} repo:{self.owner}/{self.repo}&type=issues"

        headers = self.headers
        del headers["Authorization"]

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            response.raise_for_status()
        all_issues = []
        issues_dicts = response.json()["items"]
        for issue_dict in issues_dicts:
            issue_dict["comments"] = list(self.issue_comments(issue_dict["number"]))
            issue_obj = self.transform_issue(issue_dict)
            all_issues.append(issue_obj.dict())
            if len(all_issues) >= limit:
                break
        return all_issues

    def objects(
            self,
            collection: str = None,
            object_ids: Optional[Iterable[str]] = None,
            token: str = None,
            **kwargs,
    ) -> Iterator[Dict]:
        session = self.session
        token = get_token(token)
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        if not token:
            del headers["Authorization"]
            sleep(5)
        logger.debug(f"Header: {headers}")
        params = {
            "state": "all",  # To fetch both open and closed issues and PRs
            "per_page": 100,  # Fetch 100 results per page (max allowed)
        }

        while url:
            response = session.get(url, headers=headers, params=params)
            response.raise_for_status()
            issues = response.json()
            for issue in issues:
                issue_number = issue.get("number")
                # Fetch both issue comments and PR comments
                if "pull_request" in issue:
                    issue["comments"] = list(self.pr_comments(issue_number))
                else:
                    issue["comments"] = list(self.issue_comments(issue_number))
                issue_obj = self.transform_issue(issue)
                yield issue_obj.dict()
                # Check if there are more pages to process
            url = response.links.get("next", {}).get("url")
            if not response.from_cache:
                sleep(0.2)

    def issue_comments(self, issue_number: str) -> Iterator[Dict]:
        session = self.session
        url = (
            f"https://api.github.com/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments"
        )
        params = {"per_page": 100}

        while url:
            response = session.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            yield from response.json()
            url = response.links.get("next", {}).get("url")

    def transform_issue(self, obj: Dict) -> Issue:

        if not obj.get("body"):
            obj["body"] = ""

        issue = Issue(
            id=obj.get("url"),
            number=obj.get("number"),
            title=obj.get("title"),
            user=obj.get("user").get("login"),
            labels=[label.get("name") for label in obj.get("labels")],
            state=obj.get("state"),
            assignees=[assignee.get("login") for assignee in obj.get("assignees")],
            created_at=obj.get("created_at"),
            type="pull_request" if obj.get("pull_request") else "issue",
            body=obj.get("body"),
            # pull_request=pr.get("url") if pr else None,
            comments=[
                Comment(
                    id=comment.get("url"),
                    user=comment.get("user").get("login"),
                    body=comment.get("body"),
                )
                for comment in obj.get("comments")
            ],
        )
        return issue

    def pr_comments(self, pr_number: str) -> Iterator[Dict]:
        session = self.session
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/pulls/{pr_number}/comments"
        params = {"per_page": 100}

        while url:
            response = session.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            yield from response.json()
            url = response.links.get("next", {}).get("url")
