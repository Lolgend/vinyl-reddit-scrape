"""Tests for Discogs enrichment, 5-tier search cascade, and candidate arbitration."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.discogs import DiscogsSearchCandidate
from reddit_scrape.enrichment import (
    ScoredCandidate,
    compute_cosine_similarity,
    deduplicate_candidates,
    score_candidates,
)


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







