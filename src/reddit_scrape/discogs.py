"""Discogs API HTTP client with rate limiting, adaptive backpressure, and retries."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import httpx
from typing_extensions import Self

logger = logging.getLogger(__name__)

DEFAULT_DISCOGS_BASE_URL = "https://api.discogs.com"
DEFAULT_DISCOGS_USER_AGENT = "VinylReleasesScraper/0.1.0"
DEFAULT_RATE_LIMIT_FLOOR_SECONDS = 1.15
ACCEPT_HEADER = "application/vnd.discogs.v2.plaintext+json"


class DiscogsAPIError(Exception):
    """Exception raised for Discogs API errors or network failures."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: httpx.Response | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response


@dataclass
class DiscogsSearchCandidate:
    """Search candidate returned by Discogs database search endpoint."""

    id: int
    type: str
    title: str
    master_id: int | None = None
    year: str | int | None = None
    country: str | None = None
    format: list[str] = field(default_factory=list)
    uri: str | None = None
    thumb: str | None = None
    resource_url: str | None = None


@dataclass
class DiscogsArtistCredit:
    """Artist credit entry on a Master or Release."""

    id: int
    name: str
    anv: str | None = None
    join: str | None = None
    role: str | None = None
    resource_url: str | None = None


@dataclass
class DiscogsMasterDetail:
    """Detail payload for a Discogs Master."""

    id: int
    title: str
    main_release: int | None = None
    year: int | None = None
    artists: list[DiscogsArtistCredit] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)


@dataclass
class DiscogsReleaseDetail:
    """Detail payload for a Discogs Release."""

    id: int
    title: str
    master_id: int | None = None
    year: int | None = None
    artists: list[DiscogsArtistCredit] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    country: str | None = None
    formats: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DiscogsArtistDetail:
    """Detail payload for a Discogs Artist."""

    id: int
    name: str
    realname: str | None = None
    profile: str | None = None
    resource_url: str | None = None
    uri: str | None = None
    namevariations: list[str] = field(default_factory=list)
    aliases: list[DiscogsArtistCredit] = field(default_factory=list)


class DiscogsClient:
    """Authenticated Discogs API client with rate limiting and retry handling."""

    def __init__(
        self,
        token: str | None = None,
        consumer_key: str | None = None,
        consumer_secret: str | None = None,
        user_agent: str = DEFAULT_DISCOGS_USER_AGENT,
        base_url: str = DEFAULT_DISCOGS_BASE_URL,
        rate_limit_floor_seconds: float = DEFAULT_RATE_LIMIT_FLOOR_SECONDS,
        shutdown_event: threading.Event | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.token = token.strip() if token and token.strip() else None
        self.consumer_key = consumer_key.strip() if consumer_key and consumer_key.strip() else None
        self.consumer_secret = consumer_secret.strip() if consumer_secret and consumer_secret.strip() else None
        self.user_agent = user_agent.strip() if user_agent and user_agent.strip() else DEFAULT_DISCOGS_USER_AGENT
        self.base_url = base_url.rstrip("/")
        self.rate_limit_floor_seconds = rate_limit_floor_seconds
        self.shutdown_event = shutdown_event
        self._shutdown_event = shutdown_event

        has_token = bool(self.token)
        has_key_secret = bool(self.consumer_key and self.consumer_secret)
        if not (has_token or has_key_secret):
            raise ValueError(
                "DiscogsClient requires either token or consumer_key and consumer_secret"
            )

        self._lock = threading.Lock()
        self._next_allowed_time: float = 0.0
        self._next_minimum_delay: float = self.rate_limit_floor_seconds
        self._ratelimit_remaining: int | None = None

        self._headers = self._build_headers()
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(
            headers=self._headers,
            timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=30.0),
        )

    def _build_headers(self) -> dict[str, str]:
        if self.token:
            auth = f"Discogs token={self.token}"
        else:
            auth = f"Discogs key={self.consumer_key}, secret={self.consumer_secret}"

        return {
            "Authorization": auth,
            "User-Agent": self.user_agent,
            "Accept": ACCEPT_HEADER,
        }

    def get_headers(self) -> dict[str, str]:
        """Return the default HTTP headers sent on requests."""
        return dict(self._headers)

    def _wait_interruptible(self, delay: float) -> None:
        """Wait for delay seconds or until shutdown_event is set."""
        if delay <= 0:
            return
        if self.shutdown_event is not None:
            interrupted = self.shutdown_event.wait(timeout=delay)
            if interrupted:
                raise DiscogsAPIError("Discogs request aborted due to shutdown event")
        else:
            time.sleep(delay)

    def _pace_request(self) -> None:
        """Enforce thread-safe rate limit pacing and adaptive header backpressure."""
        with self._lock:
            now = time.monotonic()
            delay_needed = max(self._next_minimum_delay, self.rate_limit_floor_seconds)
            scheduled_time = max(now, self._next_allowed_time)
            self._next_allowed_time = scheduled_time + delay_needed
            sleep_time = max(0.0, scheduled_time - now)

        if sleep_time > 0:
            self._wait_interruptible(sleep_time)

    def _handle_response_ratelimit_headers(self, response: httpx.Response) -> None:
        """Update adaptive backpressure based on X-Discogs-Ratelimit-Remaining."""
        rem_str = response.headers.get("X-Discogs-Ratelimit-Remaining")
        if rem_str is not None:
            try:
                remaining = int(rem_str)
                with self._lock:
                    self._ratelimit_remaining = remaining
                    now = time.monotonic()
                    if remaining <= 5:
                        delay = max(5.0, 60.0 / max(1, remaining))
                        self._next_minimum_delay = delay
                        self._next_allowed_time = max(self._next_allowed_time, now + delay)
                    else:
                        self._next_minimum_delay = self.rate_limit_floor_seconds
                        self._next_allowed_time = min(
                            self._next_allowed_time, now + self.rate_limit_floor_seconds
                        )
            except (ValueError, TypeError):
                pass

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Execute HTTP request with rate limit pacing, retries, and error handling."""
        url = f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"
        max_retries = 2
        attempts = 0

        while True:
            attempts += 1
            self._pace_request()

            try:
                response = self._http_client.request(method, url, params=params)
            except httpx.RequestError as exc:
                if attempts <= max_retries:
                    logger.warning(
                        "Discogs request error on %s: %s, retrying (%d/%d)...",
                        url,
                        exc,
                        attempts,
                        max_retries,
                    )
                    self._wait_interruptible(5.0)
                    continue
                raise DiscogsAPIError(f"Discogs request failed: {exc}") from exc

            self._handle_response_ratelimit_headers(response)

            if response.status_code in (429, 500, 502, 503, 504):
                if attempts <= max_retries:
                    retry_after_str = response.headers.get("Retry-After")
                    default_wait = 60.0 if response.status_code == 429 else 5.0
                    retry_wait = default_wait
                    if retry_after_str:
                        try:
                            parsed_wait = float(retry_after_str)
                            if parsed_wait >= 0:
                                retry_wait = parsed_wait
                        except (ValueError, TypeError):
                            pass
                    logger.warning(
                        "Discogs %d on %s, waiting %.1fs before retry (%d/%d)...",
                        response.status_code,
                        url,
                        retry_wait,
                        attempts,
                        max_retries,
                    )
                    self._wait_interruptible(retry_wait)
                    continue

                raise DiscogsAPIError(
                    f"Discogs API error {response.status_code} after {max_retries} retries: {response.text}",
                    status_code=response.status_code,
                    response=response,
                )

            if response.status_code >= 400:
                raise DiscogsAPIError(
                    f"Discogs API error {response.status_code}: {response.text}",
                    status_code=response.status_code,
                    response=response,
                )

            return response

    def search(
        self,
        type: str | None = None,
        artist: str | None = None,
        release_title: str | None = None,
        q: str | None = None,
        format: str | None = None,
        per_page: int = 5,
        page: int = 1,
        **kwargs: Any,
    ) -> list[DiscogsSearchCandidate]:
        """Search Discogs database with structured or keyword query."""
        params: dict[str, Any] = {"per_page": per_page, "page": page}
        if type is not None:
            params["type"] = type
        if artist is not None:
            params["artist"] = artist
        if release_title is not None:
            params["release_title"] = release_title
        if q is not None:
            params["q"] = q
        if format is not None:
            params["format"] = format
        params.update(kwargs)

        resp = self._request("GET", "/database/search", params=params)
        data = resp.json()
        candidates: list[DiscogsSearchCandidate] = []
        for item in data.get("results", []):
            candidates.append(
                DiscogsSearchCandidate(
                    id=item["id"],
                    type=item.get("type", ""),
                    title=item.get("title", ""),
                    master_id=item.get("master_id"),
                    year=item.get("year"),
                    country=item.get("country"),
                    format=item.get("format", []),
                    uri=item.get("uri"),
                    thumb=item.get("thumb"),
                    resource_url=item.get("resource_url"),
                )
            )
        return candidates

    def get_master(self, master_id: int) -> DiscogsMasterDetail:
        """Fetch details for a Discogs master release."""
        resp = self._request("GET", f"/masters/{master_id}")
        data = resp.json()
        artists: list[DiscogsArtistCredit] = []
        for art in data.get("artists", []):
            artists.append(
                DiscogsArtistCredit(
                    id=art["id"],
                    name=art.get("name", ""),
                    anv=art.get("anv"),
                    join=art.get("join"),
                    role=art.get("role"),
                    resource_url=art.get("resource_url"),
                )
            )
        return DiscogsMasterDetail(
            id=data["id"],
            title=data.get("title", ""),
            main_release=data.get("main_release"),
            year=data.get("year"),
            artists=artists,
            genres=data.get("genres", []),
            styles=data.get("styles", []),
        )

    def get_release(self, release_id: int) -> DiscogsReleaseDetail:
        """Fetch details for a Discogs release."""
        resp = self._request("GET", f"/releases/{release_id}")
        data = resp.json()
        artists: list[DiscogsArtistCredit] = []
        for art in data.get("artists", []):
            artists.append(
                DiscogsArtistCredit(
                    id=art["id"],
                    name=art.get("name", ""),
                    anv=art.get("anv"),
                    join=art.get("join"),
                    role=art.get("role"),
                    resource_url=art.get("resource_url"),
                )
            )
        return DiscogsReleaseDetail(
            id=data["id"],
            title=data.get("title", ""),
            master_id=data.get("master_id"),
            year=data.get("year"),
            artists=artists,
            genres=data.get("genres", []),
            styles=data.get("styles", []),
            country=data.get("country"),
            formats=data.get("formats", []),
        )

    def get_artist(self, artist_id: int) -> DiscogsArtistDetail:
        """Fetch details for a Discogs artist."""
        resp = self._request("GET", f"/artists/{artist_id}")
        data = resp.json()
        aliases: list[DiscogsArtistCredit] = []
        for ali in data.get("aliases", []):
            aliases.append(
                DiscogsArtistCredit(
                    id=ali["id"],
                    name=ali.get("name", ""),
                    anv=ali.get("anv"),
                    join=ali.get("join"),
                    role=ali.get("role"),
                    resource_url=ali.get("resource_url"),
                )
            )
        return DiscogsArtistDetail(
            id=data["id"],
            name=data.get("name", ""),
            realname=data.get("realname"),
            profile=data.get("profile"),
            resource_url=data.get("resource_url"),
            uri=data.get("uri"),
            namevariations=data.get("namevariations", []),
            aliases=aliases,
        )

    def close(self) -> None:
        """Close underlying HTTP client if owned."""
        if self._owns_http_client:
            self._http_client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
