"""Tests for Discogs enrichment, 5-tier search cascade, and candidate arbitration."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic_ai.models.test import TestModel
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.discogs import (
    DiscogsAPIError,
    DiscogsClient,
    DiscogsSearchCandidate,
)
from reddit_scrape.enrichment import (
    WEB_RECONCILIATION_SYSTEM_PROMPT,
    CatalogEnrichmentService,
    ScoredCandidate,
    WebReconciliationResult,
    WebSearchHit,
    build_web_reconciliation_prompt,
    compute_cosine_similarity,
    create_candidate_arbiter_agent,
    create_web_reconciliation_agent,
    deduplicate_candidates,
    get_next_queued_album,
    reconcile_web_search,
    record_enrichment_failure,
    score_candidates,
    search_web_for_release,
)
from reddit_scrape.models import Album, AlbumArtist, Artist
from reddit_scrape.persistence import compute_album_fingerprint


@pytest.fixture(scope="session")
def engine():
    try:
        eng = get_engine()
        init_db(eng)
        return eng
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Database unavailable: {exc}")


@pytest.fixture(autouse=True)
def clean_db(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE submission_snapshots, submission_releases, releases, album_artists, albums, artists, submissions CASCADE;"
            )
        )
    yield
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE submission_snapshots, submission_releases, releases, album_artists, albums, artists, submissions CASCADE;"
            )
        )


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine)



class TestCandidateScoring:
    """Tests for candidate deduplication and fastembed cosine similarity scoring."""

    def test_deduplicate_candidates_preserves_order_and_caps_at_max(self) -> None:
        raw_candidates = [
            DiscogsSearchCandidate(id=1, type="master", title="Artist - Album 1"),
            DiscogsSearchCandidate(id=2, type="master", title="Artist - Album 2"),
            DiscogsSearchCandidate(id=1, type="master", title="Artist - Album 1 Duplicate"),
            DiscogsSearchCandidate(id=3, type="release", title="Artist - Album 3"),
            DiscogsSearchCandidate(id=4, type="master", title="Artist - Album 4"),
            DiscogsSearchCandidate(id=5, type="release", title="Artist - Album 5"),
            DiscogsSearchCandidate(id=6, type="master", title="Artist - Album 6"),
        ]

        deduped = deduplicate_candidates(raw_candidates, max_candidates=5)

        assert len(deduped) == 5
        assert [c.id for c in deduped] == [1, 2, 3, 4, 5]
        # Verify deduplication key treats (type, id) uniquely
        assert deduped[0].title == "Artist - Album 1"

    def test_compute_cosine_similarity_identical_and_orthogonal(self) -> None:
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        vec_c = [0.0, 1.0, 0.0]
        vec_zero = [0.0, 0.0, 0.0]

        assert pytest.approx(compute_cosine_similarity(vec_a, vec_b), rel=1e-4) == 1.0
        assert pytest.approx(compute_cosine_similarity(vec_a, vec_c), rel=1e-4) == 0.0
        assert compute_cosine_similarity(vec_a, vec_zero) == 0.0

    def test_score_candidates_computes_similarity_and_keeps_all_candidates_no_hard_drop(
        self,
    ) -> None:
        query_text = "Converge - Jane Doe"
        candidates = [
            DiscogsSearchCandidate(
                id=101,
                type="master",
                title="Converge - Jane Doe",
                year=2001,
                country="US",
                format=["Vinyl", "LP"],
            ),
            DiscogsSearchCandidate(
                id=102,
                type="release",
                title="Converge - Jane Doe (20th Anniversary Edition)",
                master_id=101,
                year=2021,
                country="US",
                format=["Vinyl", "2xLP"],
            ),
            DiscogsSearchCandidate(
                id=999,
                type="release",
                title="Unrelated Band - Something Completely Different",
                year=1995,
                country="UK",
                format=["CD"],
            ),
        ]

        scored = score_candidates(query_text, candidates, max_candidates=5)

        assert len(scored) == 3
        # Check ranks are 1-indexed
        assert [s.search_rank for s in scored] == [1, 2, 3]

        # Exact match should have very high similarity
        assert scored[0].cosine_similarity > 0.85
        assert scored[0].id == 101
        assert scored[0].type == "master"

        # Partial match should have high similarity
        assert scored[1].cosine_similarity > 0.60
        assert scored[1].master_id == 101

        # Unrelated candidate must NOT be dropped by a hard cutoff
        assert scored[2].id == 999
        assert scored[2].cosine_similarity < scored[0].cosine_similarity

    def test_score_candidates_empty_list_returns_empty(self) -> None:
        scored = score_candidates("Some Artist - Some Album", [])
        assert scored == []


class TestArbiterPromptsAndAgent:
    """Tests for arbiter prompt construction, schema validation, and agent factory."""

    def test_build_work_arbiter_prompt_contains_all_candidate_fields(self) -> None:
        from reddit_scrape.enrichment import build_work_arbiter_prompt

        candidates = [
            ScoredCandidate(
                candidate=DiscogsSearchCandidate(
                    id=4567,
                    type="release",
                    title="Converge - Jane Doe",
                    master_id=1234,
                    year=2001,
                    country="US",
                    format=["Vinyl", "LP"],
                ),
                cosine_similarity=0.9876,
                search_rank=1,
            )
        ]

        prompt = build_work_arbiter_prompt(
            combined_artist="Converge",
            title="Jane Doe",
            tier=2,
            candidates=candidates,
        )

        assert "Converge" in prompt
        assert "Jane Doe" in prompt
        assert "Tier 2" in prompt
        assert "4567" in prompt
        assert "release" in prompt
        assert "1234" in prompt
        assert "2001" in prompt
        assert "US" in prompt
        assert "Vinyl" in prompt
        assert "0.9876" in prompt
        assert "Rank 1" in prompt or "Search Rank: 1" in prompt

    def test_build_artist_arbiter_prompt_contains_artist_metadata(self) -> None:
        from reddit_scrape.enrichment import build_artist_arbiter_prompt

        candidates = [
            ScoredCandidate(
                candidate=DiscogsSearchCandidate(
                    id=789,
                    type="artist",
                    title="Converge (2)",
                ),
                cosine_similarity=0.8888,
                search_rank=1,
            )
        ]

        prompt = build_artist_arbiter_prompt(
            artist_name="Converge",
            candidates=candidates,
        )

        assert "Target Artist: Converge" in prompt or "Converge" in prompt
        assert "Tier 5" in prompt
        assert "789" in prompt
        assert "Converge (2)" in prompt
        assert "0.8888" in prompt

    def test_create_candidate_arbiter_agent_with_test_model(self) -> None:
        from pydantic_ai.models.test import TestModel

        from reddit_scrape.enrichment import (
            CandidateArbiterResult,
            create_candidate_arbiter_agent,
        )

        custom_output = {
            "is_match": True,
            "matched_candidate_id": 4567,
            "reasoning": "Exact musical work match on Converge - Jane Doe.",
        }
        agent = create_candidate_arbiter_agent(
            model=TestModel(custom_output_args=custom_output)
        )

        result = agent.run_sync("test prompt").output
        assert isinstance(result, CandidateArbiterResult)
        assert result.is_match is True
        assert result.matched_candidate_id == 4567
        assert "Exact musical work match" in (result.reasoning or "")

    def test_create_candidate_arbiter_agent_requires_api_key_when_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reddit_scrape.enrichment import create_candidate_arbiter_agent

        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_KEY", raising=False)

        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            create_candidate_arbiter_agent(api_key=None)


class TestArbitrationAndPromotion:
    """Tests for LLM candidate arbitration and master_id promotion rules."""

    def test_arbitrate_work_candidates_promotes_master_candidate(self) -> None:
        from pydantic_ai.models.test import TestModel

        from reddit_scrape.enrichment import (
            ResolvedWorkMatch,
            arbitrate_work_candidates,
            create_candidate_arbiter_agent,
        )

        candidates = [
            ScoredCandidate(
                candidate=DiscogsSearchCandidate(
                    id=12345,
                    type="master",
                    title="Converge - Jane Doe",
                    year=2001,
                ),
                cosine_similarity=0.95,
                search_rank=1,
            )
        ]

        agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 12345,
                    "reasoning": "Direct master match",
                }
            )
        )

        match = arbitrate_work_candidates(
            combined_artist="Converge",
            title="Jane Doe",
            tier=1,
            candidates=candidates,
            arbiter_agent=agent,
        )

        assert isinstance(match, ResolvedWorkMatch)
        assert match.tier == 1
        assert match.candidate.id == 12345
        assert match.resolved_master_id == 12345
        assert match.resolved_release_id is None
        assert match.cosine_similarity == 0.95
        assert match.search_rank == 1
        assert match.reasoning == "Direct master match"

    def test_arbitrate_work_candidates_promotes_release_with_master_id(self) -> None:
        from pydantic_ai.models.test import TestModel

        from reddit_scrape.enrichment import (
            ResolvedWorkMatch,
            arbitrate_work_candidates,
            create_candidate_arbiter_agent,
        )

        candidates = [
            ScoredCandidate(
                candidate=DiscogsSearchCandidate(
                    id=8888,
                    type="release",
                    title="Converge - Jane Doe (Vinyl LP)",
                    master_id=12345,
                    year=2001,
                ),
                cosine_similarity=0.91,
                search_rank=1,
            )
        ]

        agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 8888,
                    "reasoning": "Vinyl release of master 12345",
                }
            )
        )

        match = arbitrate_work_candidates(
            combined_artist="Converge",
            title="Jane Doe",
            tier=2,
            candidates=candidates,
            arbiter_agent=agent,
        )

        assert isinstance(match, ResolvedWorkMatch)
        assert match.tier == 2
        assert match.candidate.id == 8888
        # Master ID promotion rule: promotes to master_id while retaining release_id
        assert match.resolved_master_id == 12345
        assert match.resolved_release_id == 8888
        assert match.cosine_similarity == 0.91

    def test_arbitrate_work_candidates_unmastered_release(self) -> None:
        from pydantic_ai.models.test import TestModel

        from reddit_scrape.enrichment import (
            ResolvedWorkMatch,
            arbitrate_work_candidates,
            create_candidate_arbiter_agent,
        )

        candidates = [
            ScoredCandidate(
                candidate=DiscogsSearchCandidate(
                    id=9999,
                    type="release",
                    title="Converge / Dropdead - Split 7\"",
                    master_id=None,  # Unmastered standalone release
                    year=2011,
                ),
                cosine_similarity=0.89,
                search_rank=1,
            )
        ]

        agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 9999,
                    "reasoning": "Unmastered standalone 7-inch vinyl release",
                }
            )
        )

        match = arbitrate_work_candidates(
            combined_artist="Converge / Dropdead",
            title="Split",
            tier=4,
            candidates=candidates,
            arbiter_agent=agent,
        )

        assert isinstance(match, ResolvedWorkMatch)
        assert match.tier == 4
        assert match.candidate.id == 9999
        # Unmastered rule: resolved_master_id is None, resolved_release_id is candidate.id
        assert match.resolved_master_id is None
        assert match.resolved_release_id == 9999

    def test_arbitrate_work_candidates_no_match(self) -> None:
        from pydantic_ai.models.test import TestModel

        from reddit_scrape.enrichment import (
            arbitrate_work_candidates,
            create_candidate_arbiter_agent,
        )

        candidates = [
            ScoredCandidate(
                candidate=DiscogsSearchCandidate(
                    id=111,
                    type="master",
                    title="Completely Different Artist - Different Album",
                ),
                cosine_similarity=0.22,
                search_rank=1,
            )
        ]

        agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": False,
                    "matched_candidate_id": None,
                    "reasoning": "No relevant match",
                }
            )
        )

        match = arbitrate_work_candidates(
            combined_artist="Artist X",
            title="Album Y",
            tier=1,
            candidates=candidates,
            arbiter_agent=agent,
        )

        assert match is None

    def test_arbitrate_artist_candidates_matches_constituent_artist(self) -> None:
        from pydantic_ai.models.test import TestModel

        from reddit_scrape.enrichment import (
            ResolvedArtistMatch,
            arbitrate_artist_candidates,
            create_candidate_arbiter_agent,
        )

        candidates = [
            ScoredCandidate(
                candidate=DiscogsSearchCandidate(
                    id=33333,
                    type="artist",
                    title="Converge (2)",
                ),
                cosine_similarity=0.92,
                search_rank=1,
            )
        ]

        agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 33333,
                    "reasoning": "Confirmed constituent band Converge matching homonym (2)",
                }
            )
        )

        artist_match = arbitrate_artist_candidates(
            artist_name="Converge",
            candidates=candidates,
            arbiter_agent=agent,
        )

        assert isinstance(artist_match, ResolvedArtistMatch)
        assert artist_match.artist_name == "Converge"
        assert artist_match.discogs_artist_id == 33333
        assert artist_match.candidate.id == 33333
        assert artist_match.cosine_similarity == 0.92
        assert artist_match.search_rank == 1


class TestSearchCascade:
    """Tests for the 5-tier Discogs search cascade and short-circuiting behavior."""

    def test_cascade_short_circuits_on_tier_1_match(self) -> None:
        from unittest.mock import MagicMock

        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import DiscogsClient
        from reddit_scrape.enrichment import (
            CascadeResult,
            create_candidate_arbiter_agent,
            run_discogs_search_cascade,
        )

        mock_client = MagicMock(spec=DiscogsClient)
        mock_client.search.return_value = [
            DiscogsSearchCandidate(
                id=1001,
                type="master",
                title="Converge - Jane Doe",
                year=2001,
            )
        ]

        agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 1001,
                    "reasoning": "Tier 1 master match",
                }
            )
        )

        result = run_discogs_search_cascade(
            discogs_client=mock_client,
            combined_artist="Converge",
            title="Jane Doe",
            arbiter_agent=agent,
        )

        assert isinstance(result, CascadeResult)
        assert result.work_match is not None
        assert result.work_match.tier == 1
        assert result.work_match.resolved_master_id == 1001
        assert result.work_match.resolved_release_id is None
        assert result.tier_reached == 1
        assert result.tiers_executed == [1]

        # Verify only Tier 1 was called with expected arguments
        mock_client.search.assert_called_once_with(
            type="master",
            artist="Converge",
            release_title="Jane Doe",
            per_page=5,
        )

    def test_cascade_short_circuits_on_tier_2_match_with_promotion(self) -> None:
        from unittest.mock import MagicMock

        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import DiscogsClient
        from reddit_scrape.enrichment import (
            create_candidate_arbiter_agent,
            run_discogs_search_cascade,
        )

        mock_client = MagicMock(spec=DiscogsClient)

        def mock_search(**kwargs):
            if kwargs.get("type") == "master" and "artist" in kwargs:
                # Tier 1 returns empty
                return []
            if kwargs.get("type") == "release" and "artist" in kwargs:
                # Tier 2 returns release with master_id
                return [
                    DiscogsSearchCandidate(
                        id=2002,
                        type="release",
                        title="Converge - Jane Doe (Vinyl LP)",
                        master_id=1001,
                        year=2001,
                    )
                ]
            return []

        mock_client.search.side_effect = mock_search

        agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 2002,
                    "reasoning": "Tier 2 release match",
                }
            )
        )

        result = run_discogs_search_cascade(
            discogs_client=mock_client,
            combined_artist="Converge",
            title="Jane Doe",
            arbiter_agent=agent,
        )

        assert result.work_match is not None
        assert result.work_match.tier == 2
        # Promoted master_id from candidate
        assert result.work_match.resolved_master_id == 1001
        assert result.work_match.resolved_release_id == 2002
        assert result.tiers_executed == [1, 2]
        assert mock_client.search.call_count == 2

    def test_cascade_short_circuits_on_tier_3_match(self) -> None:
        from unittest.mock import MagicMock

        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import DiscogsClient
        from reddit_scrape.enrichment import (
            create_candidate_arbiter_agent,
            run_discogs_search_cascade,
        )

        mock_client = MagicMock(spec=DiscogsClient)

        def mock_search(**kwargs):
            if kwargs.get("type") == "master" and "q" in kwargs:
                # Tier 3 keyword search matches reversed split
                return [
                    DiscogsSearchCandidate(
                        id=3003,
                        type="master",
                        title="Converge / Dropdead - Split",
                    )
                ]
            return []

        mock_client.search.side_effect = mock_search

        agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 3003,
                    "reasoning": "Tier 3 keyword match",
                }
            )
        )

        result = run_discogs_search_cascade(
            discogs_client=mock_client,
            combined_artist="Dropdead / Converge",
            title="Split",
            arbiter_agent=agent,
        )

        assert result.work_match is not None
        assert result.work_match.tier == 3
        assert result.work_match.resolved_master_id == 3003
        assert result.tiers_executed == [1, 2, 3]

    def test_cascade_short_circuits_on_tier_4_unmastered_release(self) -> None:
        from unittest.mock import MagicMock

        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import DiscogsClient
        from reddit_scrape.enrichment import (
            create_candidate_arbiter_agent,
            run_discogs_search_cascade,
        )

        mock_client = MagicMock(spec=DiscogsClient)

        def mock_search(**kwargs):
            if kwargs.get("type") == "release" and "q" in kwargs:
                # Tier 4 keyword release search
                return [
                    DiscogsSearchCandidate(
                        id=4004,
                        type="release",
                        title="Artist - Rare Single 7\"",
                        master_id=None,  # Standalone unmastered release
                    )
                ]
            return []

        mock_client.search.side_effect = mock_search

        agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 4004,
                    "reasoning": "Tier 4 unmastered release match",
                }
            )
        )

        result = run_discogs_search_cascade(
            discogs_client=mock_client,
            combined_artist="Artist",
            title="Rare Single",
            arbiter_agent=agent,
        )

        assert result.work_match is not None
        assert result.work_match.tier == 4
        assert result.work_match.resolved_master_id is None
        assert result.work_match.resolved_release_id == 4004
        assert result.tiers_executed == [1, 2, 3, 4]

    def test_cascade_falls_back_to_tier_5_skipping_already_linked_artists(self) -> None:
        from unittest.mock import MagicMock

        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import DiscogsClient
        from reddit_scrape.enrichment import (
            ConstituentArtistInfo,
            create_candidate_arbiter_agent,
            run_discogs_search_cascade,
        )

        mock_client = MagicMock(spec=DiscogsClient)

        searched_artist_queries: list[str] = []

        def mock_search(**kwargs):
            if kwargs.get("type") == "artist":
                q = kwargs.get("q", "")
                searched_artist_queries.append(q)
                if q == "Artist Unlinked":
                    return [
                        DiscogsSearchCandidate(
                            id=5005,
                            type="artist",
                            title="Artist Unlinked (2)",
                        )
                    ]
            return []

        mock_client.search.side_effect = mock_search

        agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 5005,
                    "reasoning": "Tier 5 artist match",
                }
            )
        )

        constituent_artists = [
            ConstituentArtistInfo(name="Artist Linked", discogs_artist_id=9999),
            ConstituentArtistInfo(name="Artist Unlinked", discogs_artist_id=None),
        ]

        result = run_discogs_search_cascade(
            discogs_client=mock_client,
            combined_artist="Artist Linked / Artist Unlinked",
            title="Brand New Preorder Album",
            constituent_artists=constituent_artists,
            arbiter_agent=agent,
        )

        assert result.work_match is None
        assert result.tiers_executed == [1, 2, 3, 4, 5]
        assert "Artist Unlinked" in result.artist_matches
        assert result.artist_matches["Artist Unlinked"].discogs_artist_id == 5005
        assert "Artist Linked" not in result.artist_matches

        # Verify that Artist Linked was skipped and only Artist Unlinked was queried
        assert searched_artist_queries == ["Artist Unlinked"]


class TestDetailHydration:
    """Tests for 1 follow-up Discogs Master/Release detail hydration."""

    def test_hydrate_work_detail_for_master(self) -> None:
        from unittest.mock import MagicMock

        from reddit_scrape.discogs import (
            DiscogsArtistCredit,
            DiscogsClient,
            DiscogsMasterDetail,
            DiscogsSearchCandidate,
        )
        from reddit_scrape.enrichment import (
            HydratedWorkDetail,
            ResolvedWorkMatch,
            hydrate_work_detail,
        )

        mock_client = MagicMock(spec=DiscogsClient)
        mock_client.get_master.return_value = DiscogsMasterDetail(
            id=12345,
            title="Jane Doe",
            main_release=67890,
            year=2001,
            artists=[
                DiscogsArtistCredit(id=101, name="Converge (2)", anv=None, join=None)
            ],
            genres=["Rock"],
            styles=["Hardcore"],
        )

        work_match = ResolvedWorkMatch(
            tier=1,
            candidate=DiscogsSearchCandidate(id=12345, type="master", title="Converge - Jane Doe"),
            resolved_master_id=12345,
            resolved_release_id=None,
            cosine_similarity=0.98,
            search_rank=1,
        )

        detail = hydrate_work_detail(discogs_client=mock_client, work_match=work_match)

        assert isinstance(detail, HydratedWorkDetail)
        assert detail.resolved_master_id == 12345
        assert detail.resolved_release_id == 67890
        assert detail.title == "Jane Doe"
        assert len(detail.artists) == 1
        assert detail.artists[0].id == 101
        assert detail.artists[0].name == "Converge (2)"
        mock_client.get_master.assert_called_once_with(12345)
        mock_client.get_release.assert_not_called()

    def test_hydrate_work_detail_for_unmastered_release(self) -> None:
        from unittest.mock import MagicMock

        from reddit_scrape.discogs import (
            DiscogsArtistCredit,
            DiscogsClient,
            DiscogsReleaseDetail,
            DiscogsSearchCandidate,
        )
        from reddit_scrape.enrichment import (
            HydratedWorkDetail,
            ResolvedWorkMatch,
            hydrate_work_detail,
        )

        mock_client = MagicMock(spec=DiscogsClient)
        mock_client.get_release.return_value = DiscogsReleaseDetail(
            id=55555,
            title="Converge / Dropdead Split",
            master_id=None,
            year=2011,
            artists=[
                DiscogsArtistCredit(id=101, name="Converge (2)", join="/"),
                DiscogsArtistCredit(id=202, name="Dropdead", join=None),
            ],
        )

        work_match = ResolvedWorkMatch(
            tier=4,
            candidate=DiscogsSearchCandidate(id=55555, type="release", title="Converge / Dropdead Split"),
            resolved_master_id=None,
            resolved_release_id=55555,
            cosine_similarity=0.92,
            search_rank=1,
        )

        detail = hydrate_work_detail(discogs_client=mock_client, work_match=work_match)

        assert isinstance(detail, HydratedWorkDetail)
        assert detail.resolved_master_id is None
        assert detail.resolved_release_id == 55555
        assert detail.title == "Converge / Dropdead Split"
        assert len(detail.artists) == 2
        assert detail.artists[0].name == "Converge (2)"
        assert detail.artists[0].join == "/"
        assert detail.artists[1].name == "Dropdead"
        mock_client.get_release.assert_called_once_with(55555)
        mock_client.get_master.assert_not_called()

    def test_hydrate_work_detail_preserves_existing_release_id(self) -> None:
        from unittest.mock import MagicMock

        from reddit_scrape.discogs import (
            DiscogsArtistCredit,
            DiscogsClient,
            DiscogsMasterDetail,
            DiscogsSearchCandidate,
        )
        from reddit_scrape.enrichment import (
            ResolvedWorkMatch,
            hydrate_work_detail,
        )

        mock_client = MagicMock(spec=DiscogsClient)
        mock_client.get_master.return_value = DiscogsMasterDetail(
            id=12345,
            title="Jane Doe",
            main_release=67890,
            year=2001,
            artists=[DiscogsArtistCredit(id=101, name="Converge", join=None)],
        )

        # Work match already has a resolved_release_id (e.g. from Tier 2 promotion)
        work_match = ResolvedWorkMatch(
            tier=2,
            candidate=DiscogsSearchCandidate(id=8888, type="release", title="Converge - Jane Doe (Vinyl)"),
            resolved_master_id=12345,
            resolved_release_id=8888,
            cosine_similarity=0.95,
            search_rank=1,
        )

        detail = hydrate_work_detail(discogs_client=mock_client, work_match=work_match)

        assert detail.resolved_master_id == 12345
        assert detail.resolved_release_id == 8888


class TestHomonymDisambiguationAndBilling:
    """Tests for LLM artist homonym (n) disambiguation and billing reconstruction."""

    def test_build_homonym_disambiguation_prompt_formats_artist_credits(self) -> None:
        from reddit_scrape.discogs import DiscogsArtistCredit
        from reddit_scrape.enrichment import (
            HydratedWorkDetail,
            build_homonym_disambiguation_prompt,
        )

        detail = HydratedWorkDetail(
            resolved_master_id=123,
            resolved_release_id=456,
            title="Jane Doe",
            artists=[
                DiscogsArtistCredit(id=101, name="Converge (2)", anv="Converge", join="/"),
                DiscogsArtistCredit(id=202, name="Dropdead", anv=None, join=None),
            ],
            year=2001,
        )

        prompt = build_homonym_disambiguation_prompt(detail)

        assert "Jane Doe" in prompt
        assert "101" in prompt
        assert "Converge (2)" in prompt
        assert "Join: '/'" in prompt or "join: /" in prompt.lower()
        assert "202" in prompt
        assert "Dropdead" in prompt

    def test_reconcile_artist_homonyms_strips_numeric_suffixes(self) -> None:
        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import DiscogsArtistCredit
        from reddit_scrape.enrichment import (
            CanonicalArtistCredit,
            HydratedWorkDetail,
            WorkReconciliationResult,
            create_homonym_disambiguation_agent,
            reconcile_artist_homonyms,
        )

        detail = HydratedWorkDetail(
            resolved_master_id=1001,
            resolved_release_id=2002,
            title="Bleach",
            artists=[DiscogsArtistCredit(id=999, name="Nirvana (2)", join=None)],
            year=1989,
        )

        agent = create_homonym_disambiguation_agent(
            model=TestModel(
                custom_output_args={
                    "canonical_combined_artist": "Nirvana",
                    "canonical_artists": [
                        {"discogs_artist_id": 999, "name": "Nirvana"}
                    ],
                    "canonical_album_title": "Bleach",
                    "reasoning": "Stripped Discogs numeric homonym suffix (2) from Nirvana",
                }
            )
        )

        result = reconcile_artist_homonyms(detail, disambiguation_agent=agent)

        assert isinstance(result, WorkReconciliationResult)
        assert result.canonical_combined_artist == "Nirvana"
        assert result.canonical_album_title == "Bleach"
        assert len(result.canonical_artists) == 1
        assert isinstance(result.canonical_artists[0], CanonicalArtistCredit)
        assert result.canonical_artists[0].discogs_artist_id == 999
        assert result.canonical_artists[0].name == "Nirvana"

    def test_reconcile_artist_homonyms_preserves_legitimate_parenthetical_names(self) -> None:
        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import DiscogsArtistCredit
        from reddit_scrape.enrichment import (
            HydratedWorkDetail,
            create_homonym_disambiguation_agent,
            reconcile_artist_homonyms,
        )

        detail = HydratedWorkDetail(
            resolved_master_id=5005,
            resolved_release_id=None,
            title="Monoliths & Dimensions",
            artists=[DiscogsArtistCredit(id=777, name="Sunn O)))", join=None)],
            year=2009,
        )

        agent = create_homonym_disambiguation_agent(
            model=TestModel(
                custom_output_args={
                    "canonical_combined_artist": "Sunn O)))",
                    "canonical_artists": [
                        {"discogs_artist_id": 777, "name": "Sunn O)))"}
                    ],
                    "canonical_album_title": "Monoliths & Dimensions",
                    "reasoning": "Preserved legitimate band name Sunn O))) with parentheses",
                }
            )
        )

        result = reconcile_artist_homonyms(detail, disambiguation_agent=agent)

        assert result.canonical_combined_artist == "Sunn O)))"
        assert result.canonical_artists[0].name == "Sunn O)))"
        assert result.canonical_artists[0].discogs_artist_id == 777

    def test_reconcile_artist_homonyms_multi_artist_billing_with_joins(self) -> None:
        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import DiscogsArtistCredit
        from reddit_scrape.enrichment import (
            HydratedWorkDetail,
            create_homonym_disambiguation_agent,
            reconcile_artist_homonyms,
        )

        detail = HydratedWorkDetail(
            resolved_master_id=3003,
            resolved_release_id=4004,
            title="VULTURES 1",
            artists=[
                DiscogsArtistCredit(id=111, name="Kanye West", join="&"),
                DiscogsArtistCredit(id=222, name="Ty Dolla $ign (2)", join=None),
            ],
            year=2024,
        )

        agent = create_homonym_disambiguation_agent(
            model=TestModel(
                custom_output_args={
                    "canonical_combined_artist": "Kanye West & Ty Dolla $ign",
                    "canonical_artists": [
                        {"discogs_artist_id": 111, "name": "Kanye West"},
                        {"discogs_artist_id": 222, "name": "Ty Dolla $ign"},
                    ],
                    "canonical_album_title": "VULTURES 1",
                    "reasoning": "Reconstructed Kanye West & Ty Dolla $ign billing with join '&' and stripped homonym (2)",
                }
            )
        )

        result = reconcile_artist_homonyms(detail, disambiguation_agent=agent)

        assert result.canonical_combined_artist == "Kanye West & Ty Dolla $ign"
        assert len(result.canonical_artists) == 2
        assert result.canonical_artists[0].name == "Kanye West"
        assert result.canonical_artists[1].name == "Ty Dolla $ign"
        assert result.canonical_artists[1].discogs_artist_id == 222

    def test_reconstruct_combined_artist_helper(self) -> None:
        from reddit_scrape.discogs import DiscogsArtistCredit
        from reddit_scrape.enrichment import reconstruct_combined_artist

        credits = [
            DiscogsArtistCredit(id=1, name="Artist A (2)", join="&"),
            DiscogsArtistCredit(id=2, name="Artist B", join=None),
        ]
        combined = reconstruct_combined_artist(credits, ["Artist A", "Artist B"])
        assert combined == "Artist A & Artist B"

        # Comma join delimiter
        credits_comma = [
            DiscogsArtistCredit(id=1, name="Artist A", join=","),
            DiscogsArtistCredit(id=2, name="Artist B", join=None),
        ]
        combined_comma = reconstruct_combined_artist(credits_comma, ["Artist A", "Artist B"])
        assert combined_comma == "Artist A, Artist B"

        # Missing join delimiter defaults to " / "
        credits_split = [
            DiscogsArtistCredit(id=1, name="Band 1", join=None),
            DiscogsArtistCredit(id=2, name="Band 2", join=None),
        ]
        combined_split = reconstruct_combined_artist(credits_split, ["Band 1", "Band 2"])
        assert combined_split == "Band 1 / Band 2"


class TestDetachedEnrichmentWriteBack:
    """Tests for 3-phase detached transaction write-back for matched albums."""

    def test_detached_enrichment_flow_has_zero_open_sessions_during_io_and_llm(
        self,
        engine,
        session_factory,
    ) -> None:
        from unittest.mock import MagicMock

        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import (
            DiscogsArtistCredit,
            DiscogsClient,
            DiscogsMasterDetail,
            DiscogsSearchCandidate,
        )
        from reddit_scrape.embeddings import EMBEDDING_DIM
        from reddit_scrape.enrichment import (
            CatalogEnrichmentService,
            create_candidate_arbiter_agent,
            create_homonym_disambiguation_agent,
        )
        from reddit_scrape.models import Album, AlbumArtist, Artist
        from reddit_scrape.persistence import compute_album_fingerprint

        # Seed database with a pending Album and linked Artist
        with session_factory() as session:
            artist = Artist(name="Converge (2)", discogs_artist_id=None, embedding=None)
            session.add(artist)
            session.flush()

            initial_fp = compute_album_fingerprint("Converge (2)", "Jane Doe")
            album = Album(
                combined_artist="Converge (2)",
                title="Jane Doe",
                fingerprint=initial_fp,
                discogs_status="pending",
                enrichment_attempts=1,
                enrichment_error="Previous temporary error",
            )
            session.add(album)
            session.flush()

            link = AlbumArtist(album_id=album.id, artist_id=artist.id)
            session.add(link)
            session.commit()

            target_album_id = album.id
            target_artist_id = artist.id

        # Track active database sessions to verify detached execution
        active_sessions = 0
        original_session = session_factory

        def tracked_session_factory():
            nonlocal active_sessions
            sess = original_session()
            orig_enter = sess.__enter__
            orig_exit = sess.__exit__
            orig_close = sess.close

            def wrapped_enter(*args, **kwargs):
                nonlocal active_sessions
                active_sessions += 1
                return orig_enter(*args, **kwargs)

            def wrapped_exit(*args, **kwargs):
                nonlocal active_sessions
                active_sessions -= 1
                return orig_exit(*args, **kwargs)

            def wrapped_close(*args, **kwargs):
                orig_close(*args, **kwargs)

            sess.__enter__ = wrapped_enter  # type: ignore[method-assign]
            sess.__exit__ = wrapped_exit  # type: ignore[method-assign]
            sess.close = wrapped_close  # type: ignore[method-assign]
            return sess

        # Mock Discogs client and assert zero active sessions when called
        mock_client = MagicMock(spec=DiscogsClient)

        def mock_search(**kwargs):
            assert active_sessions == 0, f"Expected 0 active sessions during Discogs search, found {active_sessions}"
            return [
                DiscogsSearchCandidate(
                    id=12345,
                    type="master",
                    title="Converge - Jane Doe",
                    year=2001,
                )
            ]

        def mock_get_master(master_id):
            assert active_sessions == 0, f"Expected 0 active sessions during Discogs detail hydration, found {active_sessions}"
            return DiscogsMasterDetail(
                id=12345,
                title="Jane Doe",
                main_release=67890,
                year=2001,
                artists=[
                    DiscogsArtistCredit(id=101, name="Converge (2)", join=None)
                ],
            )

        mock_client.search.side_effect = mock_search
        mock_client.get_master.side_effect = mock_get_master

        arbiter_agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 12345,
                    "reasoning": "Confirmed master match for Jane Doe",
                }
            )
        )

        disambiguation_agent = create_homonym_disambiguation_agent(
            model=TestModel(
                custom_output_args={
                    "canonical_combined_artist": "Converge",
                    "canonical_artists": [
                        {"discogs_artist_id": 101, "name": "Converge"}
                    ],
                    "canonical_album_title": "Jane Doe",
                    "reasoning": "Stripped (2) from Converge",
                }
            )
        )

        service = CatalogEnrichmentService(
            session_factory=tracked_session_factory,
            discogs_client=mock_client,
            arbiter_agent=arbiter_agent,
            disambiguation_agent=disambiguation_agent,
        )

        enriched_album = service.enrich_album(target_album_id)

        assert enriched_album is not None
        assert enriched_album.discogs_status == "matched"
        assert enriched_album.discogs_master_id == 12345
        assert enriched_album.discogs_release_id == 67890
        assert enriched_album.combined_artist == "Converge"
        assert enriched_album.title == "Jane Doe"
        assert enriched_album.fingerprint == compute_album_fingerprint("Converge", "Jane Doe")
        assert enriched_album.enrichment_attempts == 0
        assert enriched_album.enrichment_error is None
        assert enriched_album.last_enriched_at is not None
        assert enriched_album.embedding is not None
        assert len(enriched_album.embedding) == EMBEDDING_DIM

        # Verify constituent artist in database
        with session_factory() as session:
            reloaded_artist = session.get(Artist, target_artist_id)
            assert reloaded_artist is not None
            assert reloaded_artist.name == "Converge"
            assert reloaded_artist.discogs_artist_id == 101
            assert reloaded_artist.embedding is not None
            assert len(reloaded_artist.embedding) == EMBEDDING_DIM

    def test_detached_enrichment_unmastered_release(
        self,
        engine,
        session_factory,
    ) -> None:
        from unittest.mock import MagicMock

        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import (
            DiscogsArtistCredit,
            DiscogsClient,
            DiscogsReleaseDetail,
            DiscogsSearchCandidate,
        )
        from reddit_scrape.enrichment import (
            CatalogEnrichmentService,
            create_candidate_arbiter_agent,
            create_homonym_disambiguation_agent,
        )
        from reddit_scrape.models import Album, AlbumArtist, Artist
        from reddit_scrape.persistence import compute_album_fingerprint

        with session_factory() as session:
            artist1 = Artist(name="Converge", discogs_artist_id=None)
            artist2 = Artist(name="Dropdead", discogs_artist_id=None)
            session.add_all([artist1, artist2])
            session.flush()

            album = Album(
                combined_artist="Dropdead / Converge",
                title="Split",
                fingerprint=compute_album_fingerprint("Dropdead / Converge", "Split"),
                discogs_status="pending",
            )
            session.add(album)
            session.flush()

            session.add_all([
                AlbumArtist(album_id=album.id, artist_id=artist1.id),
                AlbumArtist(album_id=album.id, artist_id=artist2.id),
            ])
            session.commit()
            target_album_id = album.id
            art1_id = artist1.id
            art2_id = artist2.id

        mock_client = MagicMock(spec=DiscogsClient)

        def mock_search(**kwargs):
            if kwargs.get("type") == "release" and kwargs.get("format") == "Vinyl":
                return [
                    DiscogsSearchCandidate(
                        id=55555,
                        type="release",
                        title="Converge / Dropdead - Split",
                        master_id=None,  # Unmastered standalone
                    )
                ]
            return []

        mock_client.search.side_effect = mock_search
        mock_client.get_release.return_value = DiscogsReleaseDetail(
            id=55555,
            title="Split",
            master_id=None,
            year=2011,
            artists=[
                DiscogsArtistCredit(id=101, name="Converge", join="/"),
                DiscogsArtistCredit(id=202, name="Dropdead", join=None),
            ],
        )

        arbiter_agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 55555,
                    "reasoning": "Unmastered vinyl release match",
                }
            )
        )

        disambiguation_agent = create_homonym_disambiguation_agent(
            model=TestModel(
                custom_output_args={
                    "canonical_combined_artist": "Converge / Dropdead",
                    "canonical_artists": [
                        {"discogs_artist_id": 101, "name": "Converge"},
                        {"discogs_artist_id": 202, "name": "Dropdead"},
                    ],
                    "canonical_album_title": "Split",
                    "reasoning": "Standardized Converge / Dropdead split billing",
                }
            )
        )

        service = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=mock_client,
            arbiter_agent=arbiter_agent,
            disambiguation_agent=disambiguation_agent,
        )

        enriched = service.enrich_album(target_album_id)

        assert enriched is not None
        assert enriched.discogs_status == "matched"
        assert enriched.discogs_master_id is None
        assert enriched.discogs_release_id == 55555
        assert enriched.combined_artist == "Converge / Dropdead"
        assert enriched.title == "Split"

        with session_factory() as session:
            a1 = session.get(Artist, art1_id)
            a2 = session.get(Artist, art2_id)
            assert a1 is not None and a1.discogs_artist_id == 101
            assert a2 is not None and a2.discogs_artist_id == 202

    def test_detached_enrichment_multi_artist_homonyms_and_billing(
        self,
        engine,
        session_factory,
    ) -> None:
        from unittest.mock import MagicMock

        from pydantic_ai.models.test import TestModel

        from reddit_scrape.discogs import (
            DiscogsArtistCredit,
            DiscogsClient,
            DiscogsMasterDetail,
            DiscogsSearchCandidate,
        )
        from reddit_scrape.embeddings import EMBEDDING_DIM
        from reddit_scrape.enrichment import (
            CatalogEnrichmentService,
            create_candidate_arbiter_agent,
            create_homonym_disambiguation_agent,
        )
        from reddit_scrape.models import Album, AlbumArtist, Artist
        from reddit_scrape.persistence import compute_album_fingerprint

        with session_factory() as session:
            artist1 = Artist(name="Kanye West (2)", discogs_artist_id=None)
            artist2 = Artist(name="Ty Dolla $ign (3)", discogs_artist_id=None)
            session.add_all([artist1, artist2])
            session.flush()

            album = Album(
                combined_artist="Kanye West & Ty Dolla $ign",
                title="Vultures 1",
                fingerprint=compute_album_fingerprint("Kanye West & Ty Dolla $ign", "Vultures 1"),
                discogs_status="pending",
            )
            session.add(album)
            session.flush()

            session.add_all([
                AlbumArtist(album_id=album.id, artist_id=artist1.id),
                AlbumArtist(album_id=album.id, artist_id=artist2.id),
            ])
            session.commit()
            target_album_id = album.id
            art1_id = artist1.id
            art2_id = artist2.id

        mock_client = MagicMock(spec=DiscogsClient)
        mock_client.search.return_value = [
            DiscogsSearchCandidate(
                id=77777,
                type="master",
                title="Kanye West & Ty Dolla $ign - VULTURES 1",
            )
        ]
        mock_client.get_master.return_value = DiscogsMasterDetail(
            id=77777,
            title="VULTURES 1",
            main_release=88888,
            year=2024,
            artists=[
                DiscogsArtistCredit(id=111, name="Kanye West (2)", join="&"),
                DiscogsArtistCredit(id=222, name="Ty Dolla $ign (3)", join=None),
            ],
        )

        arbiter_agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 77777,
                    "reasoning": "Master match for Vultures 1",
                }
            )
        )

        disambiguation_agent = create_homonym_disambiguation_agent(
            model=TestModel(
                custom_output_args={
                    "canonical_combined_artist": "Kanye West & Ty Dolla $ign",
                    "canonical_artists": [
                        {"discogs_artist_id": 111, "name": "Kanye West"},
                        {"discogs_artist_id": 222, "name": "Ty Dolla $ign"},
                    ],
                    "canonical_album_title": "VULTURES 1",
                    "reasoning": "Cleaned homonyms and reconstructed '&' join",
                }
            )
        )

        service = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=mock_client,
            arbiter_agent=arbiter_agent,
            disambiguation_agent=disambiguation_agent,
        )

        enriched = service.enrich_album(target_album_id)

        assert enriched is not None
        assert enriched.discogs_status == "matched"
        assert enriched.discogs_master_id == 77777
        assert enriched.discogs_release_id == 88888
        assert enriched.combined_artist == "Kanye West & Ty Dolla $ign"
        assert enriched.title == "VULTURES 1"
        assert enriched.fingerprint == compute_album_fingerprint("Kanye West & Ty Dolla $ign", "VULTURES 1")

        with session_factory() as session:
            a1 = session.get(Artist, art1_id)
            a2 = session.get(Artist, art2_id)
            assert a1 is not None
            assert a1.name == "Kanye West"
            assert a1.discogs_artist_id == 111
            assert a1.embedding is not None
            assert len(a1.embedding) == EMBEDDING_DIM

            assert a2 is not None
            assert a2.name == "Ty Dolla $ign"
            assert a2.discogs_artist_id == 222
            assert a2.embedding is not None
            assert len(a2.embedding) == EMBEDDING_DIM


class TestWebSearchTool:
    """Tests for Line 3 DeepSeek native Web Search fallback tool (POST /v1/tools/search)."""

    def test_search_web_truncates_query_and_sends_required_payload(self) -> None:
        long_query = "A" * 150 + " vinyl album"
        expected_query = long_query[:128]

        captured_request: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_request["url"] = str(request.url)
            captured_request["headers"] = dict(request.headers)
            captured_request["json"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Kendrick Lamar - Untitled Album",
                            "url": "https://pitchfork.com/reviews/kendrick",
                            "summary": "Pitchfork review of the upcoming vinyl LP",
                            "snippet": "Kendrick Lamar announces upcoming 2027 vinyl release.",
                            "content": "Full article text detailing Kendrick Lamar's new work.",
                            "rerank_score": 0.94,
                        }
                    ]
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)

        hits = search_web_for_release(
            query=long_query,
            api_key="sk-test-deepseek-key",
            base_url="https://api.deepseek.com",
            http_client=client,
        )

        assert captured_request["url"] == "https://api.deepseek.com/v1/tools/search"
        assert captured_request["headers"]["authorization"] == "Bearer sk-test-deepseek-key"
        assert captured_request["json"] == {
            "query": expected_query,
            "count": 10,
            "need_content": True,
        }
        assert len(hits) == 1
        assert hits[0].title == "Kendrick Lamar - Untitled Album"
        assert hits[0].rerank_score == 0.94
        assert hits[0].url == "https://pitchfork.com/reviews/kendrick"

    def test_search_web_orders_hits_by_rerank_score_descending(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "Low Score Hit", "url": "https://example.com/1", "rerank_score": 0.45},
                        {"title": "High Score Hit", "url": "https://example.com/2", "rerank_score": 0.98},
                        {"title": "Mid Score Hit", "url": "https://example.com/3", "rerank_score": 0.72},
                    ]
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)

        hits = search_web_for_release(
            query="Converge Jane Doe vinyl",
            api_key="sk-test",
            base_url="https://api.deepseek.com",
            http_client=client,
        )

        assert len(hits) == 3
        assert [h.title for h in hits] == ["High Score Hit", "Mid Score Hit", "Low Score Hit"]
        assert [h.rerank_score for h in hits] == [0.98, 0.72, 0.45]

    def test_search_web_gracefully_handles_http_error_or_network_failure(self) -> None:
        def error_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(error_handler)
        client = httpx.Client(transport=transport)

        hits = search_web_for_release(
            query="Failing Query",
            api_key="sk-test",
            http_client=client,
        )
        assert hits == []


class TestWebReconciliationAgent:
    """Tests for DeepSeek PydanticAI web search reconciliation agent and prompts."""

    def test_web_reconciliation_prompt_and_schema_definitions(self) -> None:
        assert "canonical" in WEB_RECONCILIATION_SYSTEM_PROMPT.lower()
        assert "strip" in WEB_RECONCILIATION_SYSTEM_PROMPT.lower()
        res = WebReconciliationResult(
            canonical_combined_artist="A",
            canonical_artists=["A"],
            canonical_album_title="B",
        )
        assert res.canonical_combined_artist == "A"

    def test_build_web_reconciliation_prompt_formats_hits_and_bounds_content(self) -> None:
        hits = [
            WebSearchHit(
                title="Review: Kendrick Lamar - GNX",
                url="https://pitchfork.com/reviews/albums/kendrick-lamar-gnx/",
                summary="Pitchfork album review",
                snippet="Kendrick Lamar surprise-releases sixth studio album GNX on vinyl.",
                content="X" * 3000,
                rerank_score=0.96,
            )
        ]

        prompt = build_web_reconciliation_prompt(
            combined_artist="Kendrick Lamar",
            album_title="GNX Vinyl LP",
            hits=hits,
        )

        assert "Kendrick Lamar" in prompt
        assert "GNX Vinyl LP" in prompt
        assert "Review: Kendrick Lamar - GNX" in prompt
        assert "https://pitchfork.com/reviews/albums/kendrick-lamar-gnx/" in prompt
        assert "0.96" in prompt
        assert "X" * 1500 not in prompt

    def test_reconcile_web_search_with_test_model(self) -> None:
        hits = [
            WebSearchHit(
                title="Kendrick Lamar - GNX Official Announcement",
                url="https://example.com/gnx",
                snippet="Official album details for GNX by Kendrick Lamar.",
                rerank_score=0.99,
            )
        ]

        agent = create_web_reconciliation_agent(
            model=TestModel(
                custom_output_args={
                    "canonical_combined_artist": "Kendrick Lamar",
                    "canonical_artists": ["Kendrick Lamar"],
                    "canonical_album_title": "GNX",
                    "reasoning": "Standardized canonical title stripping vinyl descriptor",
                }
            )
        )

        result = reconcile_web_search(
            combined_artist="Kendrick Lamar",
            album_title="GNX Vinyl LP",
            hits=hits,
            web_agent=agent,
        )

        assert result.canonical_combined_artist == "Kendrick Lamar"
        assert result.canonical_artists == ["Kendrick Lamar"]
        assert result.canonical_album_title == "GNX"
        assert result.reasoning == "Standardized canonical title stripping vinyl descriptor"

    def test_reconcile_web_search_empty_hits_falls_back_gracefully(self) -> None:
        result = reconcile_web_search(
            combined_artist="Brand New Artist",
            album_title="Debut EP",
            hits=[],
            web_agent=None,
        )

        assert result.canonical_combined_artist == "Brand New Artist"
        assert result.canonical_artists == ["Brand New Artist"]
        assert result.canonical_album_title == "Debut EP"
        assert "Fallback" in (result.reasoning or "")

    def test_reconcile_web_search_handles_llm_exception_gracefully(self) -> None:
        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = RuntimeError("LLM rate limit reached")

        result = reconcile_web_search(
            combined_artist="Converge & Chelsea Wolfe",
            album_title="Bloodmoon",
            hits=[WebSearchHit(title="Search hit", url="https://example.com")],
            web_agent=mock_agent,
        )

        assert result.canonical_combined_artist == "Converge & Chelsea Wolfe"
        assert result.canonical_album_title == "Bloodmoon"
        assert "Fallback" in (result.reasoning or "")


class TestWebSearchEnrichmentStateMachine:
    """Tests for Line 3 Web Search fallback, state transitions ('partial_matched', 'not_found'), and embedding overwrites."""

    def test_unmatched_work_with_tier5_artist_match_transitions_to_partial_matched(
        self, session_factory
    ) -> None:
        with session_factory() as session:
            artist = Artist(name="Kendrick Lamar", discogs_artist_id=None)
            album = Album(
                combined_artist="Kendrick Lamar",
                title="Untitled 2027 Studio LP",
                fingerprint="fp_kendrick_lp",
                discogs_status="pending",
                embedding=[0.01] * 384,
            )
            session.add_all([artist, album])
            session.flush()
            session.add(AlbumArtist(album_id=album.id, artist_id=artist.id))
            session.commit()
            album_id = album.id
            artist_id = artist.id

        mock_discogs = MagicMock(spec=DiscogsClient)
        def search_side_effect(**kwargs: Any) -> list[DiscogsSearchCandidate]:
            search_type = kwargs.get("type")
            if search_type == "artist":
                return [
                    DiscogsSearchCandidate(
                        id=1778977,
                        type="artist",
                        title="Kendrick Lamar",
                    )
                ]
            return []

        mock_discogs.search.side_effect = search_side_effect

        arbiter_agent = create_candidate_arbiter_agent(
            model=TestModel(
                custom_output_args={
                    "is_match": True,
                    "matched_candidate_id": 1778977,
                    "reasoning": "Tier 5 artist match confirmed",
                }
            )
        )

        def mock_web_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Kendrick Lamar Announces GNX Studio Album",
                            "url": "https://example.com/gnx",
                            "snippet": "Kendrick Lamar reveals canonical title GNX",
                            "rerank_score": 0.95,
                        }
                    ]
                },
            )

        web_client = httpx.Client(transport=httpx.MockTransport(mock_web_handler))
        web_agent = create_web_reconciliation_agent(
            model=TestModel(
                custom_output_args={
                    "canonical_combined_artist": "Kendrick Lamar",
                    "canonical_artists": ["Kendrick Lamar"],
                    "canonical_album_title": "GNX",
                    "reasoning": "Standardized canonical title via web search",
                }
            )
        )

        service = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=mock_discogs,
            arbiter_agent=arbiter_agent,
            web_agent=web_agent,
            deepseek_http_client=web_client,
        )

        enriched = service.enrich_album(album_id)
        assert enriched is not None
        assert enriched.discogs_status == "partial_matched"
        assert enriched.discogs_master_id is None
        assert enriched.discogs_release_id is None
        assert enriched.combined_artist == "Kendrick Lamar"
        assert enriched.title == "GNX"
        assert enriched.fingerprint == compute_album_fingerprint("Kendrick Lamar", "GNX")
        assert len(enriched.embedding) == 384

        with session_factory() as session:
            db_artist = session.get(Artist, artist_id)
            assert db_artist is not None
            assert db_artist.discogs_artist_id == 1778977
            assert len(db_artist.embedding) == 384

    def test_unmatched_work_with_preexisting_db_artist_discogs_id_transitions_to_partial_matched(
        self, session_factory
    ) -> None:
        with session_factory() as session:
            artist = Artist(name="Converge", discogs_artist_id=292633)
            album = Album(
                combined_artist="Converge",
                title="Unreleased Tour Split 7-inch",
                fingerprint="fp_converge_split",
                discogs_status="pending",
                embedding=[0.05] * 384,
            )
            session.add_all([artist, album])
            session.flush()
            session.add(AlbumArtist(album_id=album.id, artist_id=artist.id))
            session.commit()
            album_id = album.id

        mock_discogs = MagicMock(spec=DiscogsClient)
        mock_discogs.search.return_value = []

        def mock_web_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Converge Unreleased Tour Split Vinyl",
                            "url": "https://example.com/converge-tour",
                            "rerank_score": 0.90,
                        }
                    ]
                },
            )

        web_client = httpx.Client(transport=httpx.MockTransport(mock_web_handler))
        web_agent = create_web_reconciliation_agent(
            model=TestModel(
                custom_output_args={
                    "canonical_combined_artist": "Converge",
                    "canonical_artists": ["Converge"],
                    "canonical_album_title": "Tour Split",
                    "reasoning": "Standardized title to Tour Split",
                }
            )
        )

        service = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=mock_discogs,
            web_agent=web_agent,
            deepseek_http_client=web_client,
        )

        enriched = service.enrich_album(album_id)
        assert enriched is not None
        assert enriched.discogs_status == "partial_matched"
        assert enriched.discogs_master_id is None
        assert enriched.combined_artist == "Converge"
        assert enriched.title == "Tour Split"

    def test_unmatched_work_with_zero_discogs_artist_matches_transitions_to_not_found(
        self, session_factory
    ) -> None:
        with session_factory() as session:
            artist = Artist(name="Xylophonic Nebula Syndicate", discogs_artist_id=None)
            album = Album(
                combined_artist="Xylophonic Nebula Syndicate",
                title="Quantum Groove Odyssey",
                fingerprint="fp_xns_qgo",
                discogs_status="pending",
                embedding=[0.02] * 384,
            )
            session.add_all([artist, album])
            session.flush()
            session.add(AlbumArtist(album_id=album.id, artist_id=artist.id))
            session.commit()
            album_id = album.id
            artist_id = artist.id

        mock_discogs = MagicMock(spec=DiscogsClient)
        mock_discogs.search.return_value = []

        def mock_web_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Xylophonic Nebula Syndicate - Quantum Groove Odyssey Bandcamp",
                            "url": "https://xns.bandcamp.com/album/quantum-groove-odyssey",
                            "rerank_score": 0.91,
                        }
                    ]
                },
            )

        web_client = httpx.Client(transport=httpx.MockTransport(mock_web_handler))
        web_agent = create_web_reconciliation_agent(
            model=TestModel(
                custom_output_args={
                    "canonical_combined_artist": "Xylophonic Nebula Syndicate",
                    "canonical_artists": ["Xylophonic Nebula Syndicate"],
                    "canonical_album_title": "Quantum Groove Odyssey",
                    "reasoning": "Bandcamp release canonicalization",
                }
            )
        )

        service = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=mock_discogs,
            web_agent=web_agent,
            deepseek_http_client=web_client,
        )

        enriched = service.enrich_album(album_id)
        assert enriched is not None
        assert enriched.discogs_status == "not_found"
        assert enriched.discogs_master_id is None
        assert enriched.discogs_release_id is None
        assert enriched.combined_artist == "Xylophonic Nebula Syndicate"
        assert enriched.title == "Quantum Groove Odyssey"

        with session_factory() as session:
            db_artist = session.get(Artist, artist_id)
            assert db_artist is not None
            assert db_artist.discogs_artist_id is None
            assert len(db_artist.embedding) == 384

    def test_web_search_error_falls_back_to_pass_1_names_and_completes(
        self, session_factory
    ) -> None:
        with session_factory() as session:
            artist = Artist(name="Unknown Indie Act", discogs_artist_id=None)
            album = Album(
                combined_artist="Unknown Indie Act",
                title="Self Titled EP",
                fingerprint="fp_unknown_act",
                discogs_status="pending",
                embedding=[0.01] * 384,
            )
            session.add_all([artist, album])
            session.flush()
            session.add(AlbumArtist(album_id=album.id, artist_id=artist.id))
            session.commit()
            album_id = album.id

        mock_discogs = MagicMock(spec=DiscogsClient)
        mock_discogs.search.return_value = []

        def failing_web_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        web_client = httpx.Client(transport=httpx.MockTransport(failing_web_handler))

        service = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=mock_discogs,
            deepseek_http_client=web_client,
        )

        enriched = service.enrich_album(album_id)
        assert enriched is not None
        assert enriched.discogs_status == "not_found"
        assert enriched.combined_artist == "Unknown Indie Act"
        assert enriched.title == "Self Titled EP"
        assert len(enriched.embedding) == 384


class TestPriorityQueueSelector:
    """Tests for priority queue selector idx_albums_enrichment_queue and retry state machine."""

    def test_priority_ordering_pending_p0_before_failed_p1_before_partial_matched_p2(
        self, session_factory
    ) -> None:
        now = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)

        with session_factory() as session:
            alb_p2 = Album(
                combined_artist="Artist 2",
                title="Album Partial",
                fingerprint="fp_p2",
                discogs_status="partial_matched",
                created_at=now - timedelta(days=20),
                last_enriched_at=now - timedelta(days=8),
            )
            alb_p1 = Album(
                combined_artist="Artist 1",
                title="Album Failed",
                fingerprint="fp_p1",
                discogs_status="failed",
                enrichment_attempts=1,
                created_at=now - timedelta(hours=1),
                last_enriched_at=now - timedelta(minutes=10),
            )
            alb_p0 = Album(
                combined_artist="Artist 0",
                title="Album Pending",
                fingerprint="fp_p0",
                discogs_status="pending",
                created_at=now - timedelta(minutes=5),
            )
            session.add_all([alb_p2, alb_p1, alb_p0])
            session.commit()

            # P0 must be picked first
            next_alb = get_next_queued_album(session, now=now)
            assert next_alb is not None
            assert next_alb.discogs_status == "pending"
            assert next_alb.fingerprint == "fp_p0"

            # Transition P0 out of queue
            next_alb.discogs_status = "matched"  # type: ignore[assignment]
            session.commit()

            # Next must be P1
            next_alb = get_next_queued_album(session, now=now)
            assert next_alb is not None
            assert next_alb.discogs_status == "failed"
            assert next_alb.fingerprint == "fp_p1"

            # Transition P1 out of queue
            next_alb.discogs_status = "matched"  # type: ignore[assignment]
            session.commit()

            # Next must be P2
            next_alb = get_next_queued_album(session, now=now)
            assert next_alb is not None
            assert next_alb.discogs_status == "partial_matched"
            assert next_alb.fingerprint == "fp_p2"

    def test_exponential_backoff_for_failed_albums(self, session_factory) -> None:
        now = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)

        test_cases = [
            (1, timedelta(minutes=4), False),  # 4m < 5m backoff -> not due
            (1, timedelta(minutes=6), True),   # 6m >= 5m backoff -> due
            (2, timedelta(minutes=8), False),  # 8m < 10m backoff -> not due
            (2, timedelta(minutes=11), True),  # 11m >= 10m backoff -> due
            (3, timedelta(minutes=15), False), # 15m < 20m backoff -> not due
            (3, timedelta(minutes=22), True),  # 22m >= 20m backoff -> due
            (4, timedelta(minutes=35), False), # 35m < 40m backoff -> not due
            (4, timedelta(minutes=45), True),  # 45m >= 40m backoff -> due
        ]

        for attempts, delta, should_be_due in test_cases:
            with session_factory() as session:
                alb = Album(
                    combined_artist="Fail Band",
                    title=f"Fail {attempts} {delta}",
                    fingerprint=f"fp_fail_{attempts}_{delta.total_seconds()}",
                    discogs_status="failed",
                    enrichment_attempts=attempts,
                    created_at=now - timedelta(hours=2),
                    last_enriched_at=now - delta,
                )
                session.add(alb)
                session.commit()

                res = get_next_queued_album(session, now=now)
                if should_be_due:
                    assert res is not None, f"Expected due for attempts={attempts}, delta={delta}"
                    assert res.fingerprint == alb.fingerprint
                else:
                    assert res is None, f"Expected NOT due for attempts={attempts}, delta={delta}"

                session.delete(alb)
                session.commit()

    def test_dead_letter_for_failed_at_5_attempts(self, session_factory) -> None:
        now = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)

        with session_factory() as session:
            alb = Album(
                combined_artist="Dead Letter Band",
                title="Dead Letter Album",
                fingerprint="fp_dead_letter",
                discogs_status="failed",
                enrichment_attempts=5,
                created_at=now - timedelta(days=2),
                last_enriched_at=now - timedelta(hours=5),
            )
            session.add(alb)
            session.commit()

            assert get_next_queued_album(session, now=now) is None

    def test_partial_matched_cadence_and_age_out(self, session_factory) -> None:
        now = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)

        with session_factory() as session:
            alb1 = Album(
                combined_artist="Artist PM1",
                title="Title 1",
                fingerprint="fp_pm1",
                discogs_status="partial_matched",
                created_at=now - timedelta(days=10),
                last_enriched_at=now - timedelta(days=5),
            )
            session.add(alb1)
            session.commit()
            assert get_next_queued_album(session, now=now) is None
            session.delete(alb1)
            session.commit()

            alb2 = Album(
                combined_artist="Artist PM2",
                title="Title 2",
                fingerprint="fp_pm2",
                discogs_status="partial_matched",
                created_at=now - timedelta(days=30),
                last_enriched_at=now - timedelta(days=8),
            )
            session.add(alb2)
            session.commit()
            res = get_next_queued_album(session, now=now)
            assert res is not None
            assert res.fingerprint == "fp_pm2"
            session.delete(alb2)
            session.commit()

            alb3 = Album(
                combined_artist="Artist PM3",
                title="Title 3",
                fingerprint="fp_pm3",
                discogs_status="partial_matched",
                created_at=now - timedelta(days=65),
                last_enriched_at=now - timedelta(days=10),
            )
            session.add(alb3)
            session.commit()
            assert get_next_queued_album(session, now=now) is None

    def test_matched_and_not_found_albums_not_queued(self, session_factory) -> None:
        now = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)

        with session_factory() as session:
            alb_matched = Album(
                combined_artist="Matched Band",
                title="Matched Album",
                fingerprint="fp_matched",
                discogs_status="matched",
                created_at=now - timedelta(days=5),
                last_enriched_at=now - timedelta(days=5),
            )
            alb_not_found = Album(
                combined_artist="NF Band",
                title="NF Album",
                fingerprint="fp_not_found",
                discogs_status="not_found",
                created_at=now - timedelta(days=5),
                last_enriched_at=now - timedelta(days=5),
            )
            session.add_all([alb_matched, alb_not_found])
            session.commit()

            assert get_next_queued_album(session, now=now) is None


class TestEnrichmentFailureRecording:
    """Tests for recording transient and unhandled enrichment failures with exponential backoff."""

    def test_record_failure_sets_status_failed_increments_attempts_and_sets_error(
        self, session_factory
    ) -> None:
        with session_factory() as session:
            alb = Album(
                combined_artist="Failure Act",
                title="Album Crash",
                fingerprint="fp_crash",
                discogs_status="pending",
                enrichment_attempts=0,
            )
            session.add(alb)
            session.commit()
            alb_id = alb.id

        service = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=MagicMock(spec=DiscogsClient),
        )

        service.record_failure(alb_id, "Discogs 429 Retry Exhausted")

        with session_factory() as session:
            db_alb = session.get(Album, alb_id)
            assert db_alb is not None
            assert db_alb.discogs_status == "failed"
            assert db_alb.enrichment_attempts == 1
            assert db_alb.last_enriched_at is not None
            assert "Discogs 429" in (db_alb.enrichment_error or "")

    def test_record_failure_increments_consecutive_failures(
        self, session_factory
    ) -> None:
        with session_factory() as session:
            alb = Album(
                combined_artist="Repeated Fail Act",
                title="Album Crash Repeated",
                fingerprint="fp_crash_rep",
                discogs_status="failed",
                enrichment_attempts=2,
                enrichment_error="Previous error",
            )
            session.add(alb)
            session.commit()
            alb_id = alb.id

        record_enrichment_failure(
            session_factory=session_factory,
            album_id=alb_id,
            error="Connection timeout to Discogs",
        )

        with session_factory() as session:
            db_alb = session.get(Album, alb_id)
            assert db_alb is not None
            assert db_alb.discogs_status == "failed"
            assert db_alb.enrichment_attempts == 3
            assert "Connection timeout" in (db_alb.enrichment_error or "")

    def test_enrich_album_catches_unhandled_exception_records_failure_and_returns_none(
        self, session_factory
    ) -> None:
        with session_factory() as session:
            alb = Album(
                combined_artist="Broken Act",
                title="Broken Release",
                fingerprint="fp_broken",
                discogs_status="pending",
                enrichment_attempts=0,
            )
            session.add(alb)
            session.commit()
            alb_id = alb.id

        mock_discogs = MagicMock(spec=DiscogsClient)
        mock_discogs.search.side_effect = DiscogsAPIError("502 Bad Gateway from Discogs")

        service = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=mock_discogs,
        )

        res = service.enrich_album(alb_id)
        assert res is None

        with session_factory() as session:
            db_alb = session.get(Album, alb_id)
            assert db_alb is not None
            assert db_alb.discogs_status == "failed"
            assert db_alb.enrichment_attempts == 1
            assert "502 Bad Gateway" in (db_alb.enrichment_error or "")

    def test_enrich_album_with_raise_on_error_records_failure_then_raises(
        self, session_factory
    ) -> None:
        with session_factory() as session:
            alb = Album(
                combined_artist="Fatal Act",
                title="Fatal Release",
                fingerprint="fp_fatal",
                discogs_status="pending",
                enrichment_attempts=0,
            )
            session.add(alb)
            session.commit()
            alb_id = alb.id

        mock_discogs = MagicMock(spec=DiscogsClient)
        mock_discogs.search.side_effect = RuntimeError("Fatal LLM quota exhausted")

        service = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=mock_discogs,
        )

        with pytest.raises(RuntimeError, match="Fatal LLM quota"):
            service.enrich_album(alb_id, raise_on_error=True)

        with session_factory() as session:
            db_alb = session.get(Album, alb_id)
            assert db_alb is not None
            assert db_alb.discogs_status == "failed"
            assert db_alb.enrichment_attempts == 1
            assert "Fatal LLM quota" in (db_alb.enrichment_error or "")








