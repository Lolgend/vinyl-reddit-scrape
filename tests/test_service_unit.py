"""Unit tests for PollerService wiring, lifecycle, and candidate retriever integration."""

from unittest.mock import AsyncMock, MagicMock

from reddit_scrape.extraction import (
    BatchExtractedItem,
    BatchExtractionResult,
    ExtractedRelease,
    ExtractionWorker,
)
from reddit_scrape.persistence import (
    CatalogCandidates,
    CatalogReleaseCandidate,
    ReleasePersistenceService,
)
from reddit_scrape.service import PollerService


class TestPollerServiceWiringUnit:
    """Test component initialization and candidate retriever wiring on PollerService."""

    def test_poller_service_wires_candidate_retriever_by_default(self):
        mock_factory = MagicMock()
        service = PollerService(session_factory=mock_factory)

        assert isinstance(service.persistence_service, ReleasePersistenceService)
        assert isinstance(service.extraction_worker, ExtractionWorker)
        assert (
            service.extraction_worker.candidate_retriever
            == service.persistence_service.find_candidates
        )
        service.stop()

    def test_poller_service_wires_candidate_retriever_when_worker_has_none(self):
        mock_factory = MagicMock()
        worker = ExtractionWorker()
        assert worker.candidate_retriever is None

        service = PollerService(session_factory=mock_factory, extraction_worker=worker)
        assert (
            worker.candidate_retriever
            == service.persistence_service.find_candidates
        )
        service.stop()

    def test_poller_service_preserves_custom_candidate_retriever(self):
        mock_factory = MagicMock()
        custom_retriever = MagicMock()
        worker = ExtractionWorker(candidate_retriever=custom_retriever)

        service = PollerService(session_factory=mock_factory, extraction_worker=worker)
        assert worker.candidate_retriever == custom_retriever
        service.stop()


class TestPollerServiceMultiCycleUnit:
    """Test multi-cycle candidate retrieval and canonical alignment orchestration without live DB."""

    def test_multi_cycle_mocked_extraction_receives_candidates_and_aligns(self):
        mock_factory = MagicMock()
        mock_persistence = MagicMock(spec=ReleasePersistenceService)

        cycle_1_candidates = CatalogCandidates(releases=[], artists=[])
        cycle_2_candidates = CatalogCandidates(
            releases=[
                CatalogReleaseCandidate(
                    id="rel-cycle-1",
                    combined_artist="Converge / Dropdead",
                    album_title='Split 7"',
                    artists=["Converge", "Dropdead"],
                    variant="Opaque Emerald LP",
                    retailer="Deathwish",
                    similarity=0.92,
                )
            ],
            artists=["Converge", "Dropdead"],
        )

        def mock_find_candidates(query_titles, top_k=5, min_similarity=0.50):
            titles = [query_titles] if isinstance(query_titles, str) else list(query_titles)
            if any("Converge / Dropdead - Split" in t for t in titles):
                return cycle_1_candidates
            return cycle_2_candidates

        mock_persistence.find_candidates.side_effect = mock_find_candidates

        captured_prompts: list[str] = []
        batch_agent = MagicMock()

        async def fake_agent_run(prompt: str):
            captured_prompts.append(prompt)
            if "sub_1" in prompt or "Converge / Dropdead - Split" in prompt:
                return MagicMock(
                    output=BatchExtractionResult(
                        items=[
                            BatchExtractedItem(
                                submission_id="t3_sub_1",
                                releases=[
                                    ExtractedRelease(
                                        artists=["Converge", "Dropdead"],
                                        combined_artist="Converge / Dropdead",
                                        album_title='Split 7"',
                                        variant="Opaque Emerald LP",
                                        retailer="Deathwish",
                                    )
                                ],
                            )
                        ]
                    )
                )
            elif "sub_2" in prompt:
                # Cycle 2: reversed title and artist order aligned to canonical
                return MagicMock(
                    output=BatchExtractionResult(
                        items=[
                            BatchExtractedItem(
                                submission_id="t3_sub_2",
                                releases=[
                                    ExtractedRelease(
                                        artists=["Converge", "Dropdead"],
                                        combined_artist="Converge / Dropdead",
                                        album_title='Split 7"',
                                        variant="Opaque Emerald LP",
                                        retailer="Deathwish",
                                    )
                                ],
                            )
                        ]
                    )
                )
            else:
                # Cycle 3: new variant of Cycle 1 album
                return MagicMock(
                    output=BatchExtractionResult(
                        items=[
                            BatchExtractedItem(
                                submission_id="t3_sub_3",
                                releases=[
                                    ExtractedRelease(
                                        artists=["Converge", "Dropdead"],
                                        combined_artist="Converge / Dropdead",
                                        album_title='Split 7"',
                                        variant="Pink /500",
                                        retailer="Deathwish",
                                    )
                                ],
                            )
                        ]
                    )
                )

        batch_agent.run = AsyncMock(side_effect=fake_agent_run)
        worker = ExtractionWorker(batch_agent=batch_agent)

        service = PollerService(
            session_factory=mock_factory,
            persistence_service=mock_persistence,
            extraction_worker=worker,
        )

        assert worker.candidate_retriever == mock_persistence.find_candidates

        # --- Cycle 1 ---
        mock_session_1 = MagicMock()
        mock_session_1.execute.return_value.all.return_value = [
            ("t3_sub_1", 'Converge / Dropdead - Split 7" (Opaque Emerald LP) [Deathwish]', "New Release")
        ]
        mock_factory.return_value.__enter__.return_value = mock_session_1

        c1_count = service.process_unextracted_submissions()
        assert c1_count == 1
        assert len(captured_prompts) == 1
        assert "Existing Catalog Reference:" not in captured_prompts[0]
        mock_persistence.persist.assert_called_once()
        _, kwargs_1 = mock_persistence.persist.call_args
        assert kwargs_1["submission_id"] == "t3_sub_1"
        assert kwargs_1["releases"].releases[0].album_title == 'Split 7"'

        # --- Cycle 2 ---
        mock_session_2 = MagicMock()
        mock_session_2.execute.return_value.all.return_value = [
            ("t3_sub_2", 'Split 7" - Dropdead / Converge (Opaque Emerald LP) [Deathwish]', "Pre-Order")
        ]
        mock_factory.return_value.__enter__.return_value = mock_session_2
        mock_persistence.persist.reset_mock()

        c2_count = service.process_unextracted_submissions()
        assert c2_count == 1
        assert len(captured_prompts) == 2
        assert "Existing Catalog Reference:" in captured_prompts[1]
        assert "Combined Artist: Converge / Dropdead" in captured_prompts[1]
        assert 'Album Title: Split 7"' in captured_prompts[1]
        assert "Canonical Artists:" in captured_prompts[1]
        assert "- Converge" in captured_prompts[1]
        assert "- Dropdead" in captured_prompts[1]

        mock_persistence.persist.assert_called_once()
        _, kwargs_2 = mock_persistence.persist.call_args
        assert kwargs_2["submission_id"] == "t3_sub_2"
        # Canonical alignment verified on extracted release attributes
        c2_rel = kwargs_2["releases"].releases[0]
        assert c2_rel.combined_artist == "Converge / Dropdead"
        assert c2_rel.artists == ["Converge", "Dropdead"]
        assert c2_rel.album_title == 'Split 7"'
        assert c2_rel.variant == "Opaque Emerald LP"

        # --- Cycle 3 (New Variant of Cycle 1 Album) ---
        mock_session_3 = MagicMock()
        mock_session_3.execute.return_value.all.return_value = [
            ("t3_sub_3", 'Split 7" - Dropdead / Converge (Pink /500) [Deathwish]', None)
        ]
        mock_factory.return_value.__enter__.return_value = mock_session_3
        mock_persistence.persist.reset_mock()

        c3_count = service.process_unextracted_submissions()
        assert c3_count == 1
        assert len(captured_prompts) == 3
        assert "Existing Catalog Reference:" in captured_prompts[2]

        mock_persistence.persist.assert_called_once()
        _, kwargs_3 = mock_persistence.persist.call_args
        assert kwargs_3["submission_id"] == "t3_sub_3"
        c3_rel = kwargs_3["releases"].releases[0]
        assert c3_rel.combined_artist == "Converge / Dropdead"
        assert c3_rel.artists == ["Converge", "Dropdead"]
        assert c3_rel.album_title == 'Split 7"'
        assert c3_rel.variant == "Pink /500"

        service.stop()
