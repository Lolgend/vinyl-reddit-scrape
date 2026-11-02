"""Tests for DiscogsClient authentication, rate limiting, and endpoints."""

import threading
import time
from unittest.mock import patch

import httpx
import pytest

from reddit_scrape.discogs import (
    DiscogsAPIError,
    DiscogsArtistDetail,
    DiscogsClient,
    DiscogsMasterDetail,
    DiscogsReleaseDetail,
    DiscogsSearchCandidate,
)


class TestDiscogsAuthenticationAndHeaders:
    """Tests for DiscogsClient credentials and request headers."""

    def test_client_token_auth_headers(self):
        client = DiscogsClient(token="my_token_abc")
        headers = client.get_headers()
        assert headers["Authorization"] == "Discogs token=my_token_abc"
        assert headers["User-Agent"] == "VinylReleasesScraper/0.1.0"
        assert headers["Accept"] == "application/vnd.discogs.v2.plaintext+json"

    def test_client_key_secret_auth_headers(self):
        client = DiscogsClient(
            consumer_key="key_123",
            consumer_secret="secret_456",
        )
        headers = client.get_headers()
        assert headers["Authorization"] == "Discogs key=key_123, secret=secret_456"
        assert headers["User-Agent"] == "VinylReleasesScraper/0.1.0"
        assert headers["Accept"] == "application/vnd.discogs.v2.plaintext+json"

    def test_client_prefers_token_over_key_secret(self):
        client = DiscogsClient(
            token="preferred_token",
            consumer_key="key_123",
            consumer_secret="secret_456",
        )
        headers = client.get_headers()
        assert headers["Authorization"] == "Discogs token=preferred_token"

    def test_client_missing_credentials_raises(self):
        with pytest.raises(ValueError, match="DiscogsClient requires either token or consumer_key and consumer_secret"):
            DiscogsClient()

    def test_client_partial_key_secret_raises(self):
        with pytest.raises(ValueError, match="DiscogsClient requires either token or consumer_key and consumer_secret"):
            DiscogsClient(consumer_key="only_key")

        with pytest.raises(ValueError, match="DiscogsClient requires either token or consumer_key and consumer_secret"):
            DiscogsClient(consumer_secret="only_secret")

    def test_client_custom_user_agent(self):
        client = DiscogsClient(
            token="my_token",
            user_agent="MyCustomApp/2.1.0",
        )
        headers = client.get_headers()
        assert headers["User-Agent"] == "MyCustomApp/2.1.0"


class TestDiscogsEndpoints:
    """Tests for DiscogsClient API endpoints and typed response parsing."""

    def test_search_candidates_parsing(self):
        sample_results = {
            "results": [
                {
                    "id": 111,
                    "type": "master",
                    "title": "Converge - Jane Doe",
                    "master_id": 111,
                    "year": "2001",
                    "country": "US",
                    "format": ["Vinyl", "LP", "Album"],
                    "uri": "/master/111-Converge-Jane-Doe",
                    "thumb": "https://img.discogs.com/thumb.jpg",
                    "resource_url": "https://api.discogs.com/masters/111",
                },
                {
                    "id": 222,
                    "type": "release",
                    "title": "Converge - Jane Doe (20th Anniversary)",
                    "master_id": 111,
                    "year": "2021",
                    "country": "US",
                    "format": ["Vinyl", "2xLP"],
                    "uri": "/release/222",
                },
            ]
        }

        def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/database/search"
            assert request.url.params["artist"] == "Converge"
            assert request.url.params["release_title"] == "Jane Doe"
            assert request.url.params["type"] == "master"
            assert request.url.params["per_page"] == "5"
            return httpx.Response(200, json=sample_results)

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(token="token_123", http_client=http_client)

        candidates = client.search(
            type="master",
            artist="Converge",
            release_title="Jane Doe",
            per_page=5,
        )

        assert len(candidates) == 2
        c1 = candidates[0]
        assert isinstance(c1, DiscogsSearchCandidate)
        assert c1.id == 111
        assert c1.type == "master"
        assert c1.title == "Converge - Jane Doe"
        assert c1.master_id == 111
        assert c1.year == "2001"
        assert c1.format == ["Vinyl", "LP", "Album"]

        c2 = candidates[1]
        assert c2.id == 222
        assert c2.type == "release"
        assert c2.master_id == 111

    def test_get_master_detail_parsing(self):
        sample_master = {
            "id": 12345,
            "title": "Jane Doe",
            "main_release": 99999,
            "year": 2001,
            "artists": [
                {
                    "id": 100,
                    "name": "Converge",
                    "anv": "",
                    "join": "",
                    "role": "",
                    "resource_url": "https://api.discogs.com/artists/100",
                }
            ],
            "genres": ["Rock"],
            "styles": ["Hardcore", "Mathcore"],
        }

        def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/masters/12345"
            return httpx.Response(200, json=sample_master)

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(token="token_123", http_client=http_client)

        master = client.get_master(12345)
        assert isinstance(master, DiscogsMasterDetail)
        assert master.id == 12345
        assert master.title == "Jane Doe"
        assert master.main_release == 99999
        assert master.year == 2001
        assert len(master.artists) == 1
        assert master.artists[0].name == "Converge"
        assert master.genres == ["Rock"]

    def test_get_release_detail_parsing(self):
        sample_release = {
            "id": 67890,
            "title": "Split 7\"",
            "master_id": 54321,
            "year": 2011,
            "artists": [
                {
                    "id": 100,
                    "name": "Converge",
                    "anv": "",
                    "join": "/",
                    "role": "",
                },
                {
                    "id": 200,
                    "name": "Dropdead",
                    "anv": "",
                    "join": "",
                    "role": "",
                },
            ],
            "country": "US",
            "genres": ["Rock"],
            "styles": ["Hardcore", "Grindcore"],
            "formats": [{"name": "Vinyl", "qty": "1", "descriptions": ["7\"", "45 RPM"]}],
        }

        def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/releases/67890"
            return httpx.Response(200, json=sample_release)

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(token="token_123", http_client=http_client)

        release = client.get_release(67890)
        assert isinstance(release, DiscogsReleaseDetail)
        assert release.id == 67890
        assert release.master_id == 54321
        assert len(release.artists) == 2
        assert release.artists[0].name == "Converge"
        assert release.artists[0].join == "/"
        assert release.artists[1].name == "Dropdead"
        assert len(release.formats) == 1

    def test_get_artist_detail_parsing(self):
        sample_artist = {
            "id": 300,
            "name": "Nirvana (2)",
            "realname": "Kurt Cobain, Krist Novoselic, Dave Grohl",
            "profile": "American grunge band formed in 1987 in Aberdeen, Washington.",
            "namevariations": ["Nirvana US"],
            "aliases": [{"id": 301, "name": "Fecal Matter"}],
        }

        def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/artists/300"
            return httpx.Response(200, json=sample_artist)

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(token="token_123", http_client=http_client)

        artist = client.get_artist(300)
        assert isinstance(artist, DiscogsArtistDetail)
        assert artist.id == 300
        assert artist.name == "Nirvana (2)"
        assert artist.realname == "Kurt Cobain, Krist Novoselic, Dave Grohl"
        assert artist.namevariations == ["Nirvana US"]
        assert len(artist.aliases) == 1
        assert artist.aliases[0].name == "Fecal Matter"

    def test_non_retryable_404_raises_discogs_api_error(self):
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"message": "Release not found."})

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(token="token_123", http_client=http_client)

        with pytest.raises(DiscogsAPIError) as exc_info:
            client.get_release(99999999)
        assert exc_info.value.status_code == 404


class TestDiscogsRateLimitingAndBackpressure:
    """Tests for thread-safe rate limiting, adaptive backpressure, and shutdown interruption."""

    def test_rate_limiter_enforces_monotonic_floor(self):
        timeline = [100.0, 100.10, 100.20, 101.25]
        timeline_idx = 0

        def fake_monotonic():
            nonlocal timeline_idx
            val = timeline[min(timeline_idx, len(timeline) - 1)]
            timeline_idx += 1
            return val

        sleeps: list[float] = []

        def fake_sleep(dur: float):
            sleeps.append(dur)

        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": []})

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)

        with patch("time.monotonic", side_effect=fake_monotonic), patch("time.sleep", side_effect=fake_sleep):
            client = DiscogsClient(token="token_123", http_client=http_client)
            # Call 1: finishes at 100.10
            client.search(type="master", q="Test 1")
            # Call 2: arrives at 100.20 (0.10s after call 1 finished) -> sleeps 1.15 - 0.10 = 1.05s
            client.search(type="master", q="Test 2")

        assert len(sleeps) == 1
        assert pytest.approx(sleeps[0], rel=1e-2) == 1.05

    def test_adaptive_backpressure_stretches_delay_when_remaining_le_5(self):
        # 1st response has X-Discogs-Ratelimit-Remaining: 4
        # Delay should be stretched to max(5.0, 60.0 / 4) = 15.0 seconds
        timeline = [100.0, 100.0, 101.0, 101.0]
        timeline_idx = 0

        def fake_monotonic():
            nonlocal timeline_idx
            val = timeline[min(timeline_idx, len(timeline) - 1)]
            timeline_idx += 1
            return val

        sleeps: list[float] = []

        def fake_sleep(dur: float):
            sleeps.append(dur)

        responses = [
            httpx.Response(200, json={"results": []}, headers={"X-Discogs-Ratelimit-Remaining": "4"}),
            httpx.Response(200, json={"results": []}, headers={"X-Discogs-Ratelimit-Remaining": "3"}),
        ]
        call_count = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)

        with patch("time.monotonic", side_effect=fake_monotonic), patch("time.sleep", side_effect=fake_sleep):
            client = DiscogsClient(token="token_123", http_client=http_client)
            client.search(type="master", q="Test 1")
            # At this point, ratelimit remaining = 4, so next min delay is 15.0s
            client.search(type="master", q="Test 2")

        assert len(sleeps) == 1
        # Elapsed was 101.0 - 100.0 = 1.0s, delay needed was 15.0s -> sleep 14.0s
        assert pytest.approx(sleeps[0], rel=1e-2) == 14.0

    def test_adaptive_backpressure_resets_when_remaining_gt_5(self):
        timeline = [100.0, 100.0, 101.0, 101.0, 102.0, 102.0]
        timeline_idx = 0

        def fake_monotonic():
            nonlocal timeline_idx
            val = timeline[min(timeline_idx, len(timeline) - 1)]
            timeline_idx += 1
            return val

        sleeps: list[float] = []

        def fake_sleep(dur: float):
            sleeps.append(dur)

        responses = [
            httpx.Response(200, json={"results": []}, headers={"X-Discogs-Ratelimit-Remaining": "2"}),
            httpx.Response(200, json={"results": []}, headers={"X-Discogs-Ratelimit-Remaining": "50"}),
            httpx.Response(200, json={"results": []}),
        ]
        call_count = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)

        with patch("time.monotonic", side_effect=fake_monotonic), patch("time.sleep", side_effect=fake_sleep):
            client = DiscogsClient(token="token_123", http_client=http_client)
            client.search(type="master", q="Test 1")
            # Call 2: backpressure stretched to max(5.0, 60/2) = 30.0s
            client.search(type="master", q="Test 2")
            # Call 3: remaining was 50 (>5), so reset to 1.15s
            client.search(type="master", q="Test 3")

        assert len(sleeps) == 2
        # First sleep was stretched
        assert pytest.approx(sleeps[0], rel=1e-2) == 29.0
        # Second sleep was back to floor (1.15 - 1.0 = 0.15)
        assert pytest.approx(sleeps[1], rel=1e-2) == 0.15

    def test_shutdown_event_interrupts_rate_limit_sleep(self):
        shutdown_event = threading.Event()
        shutdown_event.set()  # Already signaled

        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": []})

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(
            token="token_123",
            shutdown_event=shutdown_event,
            http_client=http_client,
        )
        # Pre-set next allowed time to future so delay is needed
        client._next_allowed_time = time.monotonic() + 10.0

        with pytest.raises(DiscogsAPIError, match="aborted due to shutdown event"):
            client.search(type="master", q="Test")

    def test_rate_limiter_thread_safety(self):
        call_times: list[float] = []
        lock = threading.Lock()

        def mock_handler(request: httpx.Request) -> httpx.Response:
            with lock:
                call_times.append(time.monotonic())
            return httpx.Response(200, json={"results": []})

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        # Use small floor like 0.05s to test real concurrent timing quickly
        client = DiscogsClient(
            token="token_123",
            rate_limit_floor_seconds=0.05,
            http_client=http_client,
        )

        threads = [
            threading.Thread(target=client.search, kwargs={"type": "master", "q": f"Test {i}"})
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(call_times) == 4
        # Sort call times and verify spacing between consecutive requests is >= 0.045s
        call_times.sort()
        for i in range(len(call_times) - 1):
            interval = call_times[i + 1] - call_times[i]
            assert interval >= 0.04, f"Interval between call {i} and {i+1} was {interval:.4f}s (< 0.04s)"


class TestDiscogsRetries:
    """Tests for HTTP 429 and 5xx retries and Retry-After handling."""

    def test_429_retries_with_retry_after_and_recovers(self):
        retry_waits: list[float] = []

        def fake_wait(delay: float):
            retry_waits.append(delay)

        attempts = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"Retry-After": "10"}, text="Rate limit exceeded")
            return httpx.Response(200, json={"id": 1, "title": "Success", "artists": []})

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(token="token_123", http_client=http_client)

        with patch.object(client, "_wait_interruptible", side_effect=fake_wait):
            master = client.get_master(1)

        assert master.title == "Success"
        assert attempts == 2
        assert 10.0 in retry_waits

    def test_429_exhaustion_raises_api_error(self):
        attempts = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(429, headers={"Retry-After": "2"}, text="Still limited")

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(token="token_123", http_client=http_client)

        with (
            patch.object(client, "_wait_interruptible"),
            pytest.raises(DiscogsAPIError) as exc_info,
        ):
            client.get_master(1)

        assert attempts == 3  # Initial + 2 retries
        assert exc_info.value.status_code == 429

    def test_5xx_retries_with_default_5s_and_recovers(self):
        retry_waits: list[float] = []

        def fake_wait(delay: float):
            retry_waits.append(delay)

        attempts = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                # 502 with no Retry-After -> should default to 5.0s
                return httpx.Response(502, text="Bad Gateway")
            return httpx.Response(200, json={"id": 2, "title": "Recovered", "artists": []})

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(token="token_123", http_client=http_client)

        with patch.object(client, "_wait_interruptible", side_effect=fake_wait):
            release = client.get_release(2)

        assert release.title == "Recovered"
        assert attempts == 2
        assert 5.0 in retry_waits

    def test_5xx_exhaustion_raises_api_error(self):
        attempts = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(503, text="Service Unavailable")

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(token="token_123", http_client=http_client)

        with (
            patch.object(client, "_wait_interruptible"),
            pytest.raises(DiscogsAPIError) as exc_info,
        ):
            client.get_release(2)

        assert attempts == 3
        assert exc_info.value.status_code == 503

    def test_shutdown_event_interrupts_retry_wait(self):
        shutdown_event = threading.Event()

        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "60"}, text="Rate limit")

        transport = httpx.MockTransport(mock_handler)
        http_client = httpx.Client(transport=transport)
        client = DiscogsClient(
            token="token_123",
            shutdown_event=shutdown_event,
            http_client=http_client,
        )

        # Trigger shutdown during wait
        original_wait = client._wait_interruptible

        def trigger_shutdown_wait(delay: float):
            shutdown_event.set()
            original_wait(delay)

        with (
            patch.object(client, "_wait_interruptible", side_effect=trigger_shutdown_wait),
            pytest.raises(DiscogsAPIError, match="aborted due to shutdown event"),
        ):
            client.get_artist(10)


