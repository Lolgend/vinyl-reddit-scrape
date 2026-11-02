"""Tests for Discogs enrichment, 5-tier search cascade, and candidate arbitration."""

from __future__ import annotations

import pytest

from reddit_scrape.discogs import DiscogsSearchCandidate
from reddit_scrape.enrichment import (
    ScoredCandidate,
    compute_cosine_similarity,
    deduplicate_candidates,
    score_candidates,
)


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



