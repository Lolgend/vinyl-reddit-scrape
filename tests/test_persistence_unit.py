import hashlib
import uuid
from unittest.mock import MagicMock, patch

from sqlalchemy.dialects import postgresql

from reddit_scrape.extraction import ExtractedRelease
from reddit_scrape.models import Album, Artist, Release, Submission
from reddit_scrape.persistence import (
    STANDARD_TOKEN,
    CatalogCandidates,
    CatalogReleaseCandidate,
    ReleasePersistenceService,
    compute_album_fingerprint,
    compute_release_fingerprint,
    find_catalog_candidates,
    persist_extracted_releases,
)


class TestFingerprintsUnit:
    def test_compute_album_fingerprint_normalization_and_digest(self):
        fp1 = compute_album_fingerprint("  Daft   Punk  ", "  Discovery  ")
        fp2 = compute_album_fingerprint("daft punk", "discovery")
        assert fp1 == fp2
        assert len(fp1) == 64
        expected = hashlib.sha256(b"daft punk|discovery").hexdigest()
        assert fp1 == expected

    def test_compute_album_fingerprint_unicode_dashes_and_punctuation(self):
        fp1 = compute_album_fingerprint("Artist", "Album \u2013 Part 1")
        fp2 = compute_album_fingerprint("Artist", "Album - Part 1")
        assert fp1 == fp2

        fp_split1 = compute_album_fingerprint("Converge / Dropdead", "Split")
        fp_split2 = compute_album_fingerprint("Converge Dropdead", "Split")
        assert fp_split1 != fp_split2

    def test_compute_release_fingerprint_scoped_to_album_id(self):
        album_id_str = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
        album_id_uuid = uuid.UUID(album_id_str)

        fp_uuid = compute_release_fingerprint(album_id_uuid, variant=None, retailer=None)
        fp_str = compute_release_fingerprint(album_id_str, variant="", retailer="  ")
        assert fp_uuid == fp_str
        assert len(fp_uuid) == 64

        expected = hashlib.sha256(f"{album_id_str}|{STANDARD_TOKEN}|{STANDARD_TOKEN}".encode()).hexdigest()
        assert fp_uuid == expected

    def test_compute_release_fingerprint_standard_vs_variant_vs_retailer(self):
        album_id = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
        fp_std = compute_release_fingerprint(album_id, None, None)
        fp_var = compute_release_fingerprint(album_id, "Red Vinyl", None)
        fp_ret = compute_release_fingerprint(album_id, None, "Rough Trade")
        fp_both = compute_release_fingerprint(album_id, "Red Vinyl", "Rough Trade")

        assert len({fp_std, fp_var, fp_ret, fp_both}) == 4

    def test_compute_release_fingerprint_different_albums_same_variant(self):
        album_1 = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
        album_2 = "b1ffcd00-0d1c-4ef8-bb6d-6bb9bd380a22"
        fp1 = compute_release_fingerprint(album_1, "Red Vinyl", "Rough Trade")
        fp2 = compute_release_fingerprint(album_2, "Red Vinyl", "Rough Trade")
        assert fp1 != fp2


class TestPersistenceEmbeddingsUnit:
    def test_persist_single_artist_generates_and_stores_embeddings(self):
        mock_session = MagicMock()
        mock_sub = MagicMock(spec=Submission)
        mock_sub.id = "sub123"
        mock_session.get.return_value = mock_sub

        mock_release = MagicMock(spec=Release)
        mock_release.id = "rel-uuid-1"

        # Mock execute returning scalar_one for album, artist, and release inserts
        mock_session.execute.return_value.scalar_one.side_effect = ["album-uuid-1", "art-uuid-1", "rel-uuid-1"]

        def get_side_effect(model_cls, pk):
            if model_cls is Submission:
                return mock_sub
            if model_cls is Release:
                return mock_release
            return None

        mock_session.get.side_effect = get_side_effect

        ext = ExtractedRelease(
            artists=["Daft Punk"],
            combined_artist="Daft Punk",
            album_title="Discovery",
            variant="2LP Gatefold",
            retailer="Rough Trade",
        )

        dummy_vec_album = [0.1] * 384
        dummy_vec_art = [0.2] * 384

        def fake_embed(texts):
            result = []
            for t in texts:
                if "Discovery" in t:
                    result.append(dummy_vec_album)
                else:
                    result.append(dummy_vec_art)
            return result

        with patch("reddit_scrape.persistence.embed_texts") as mock_embed:
            mock_embed.side_effect = fake_embed

            results = persist_extracted_releases(mock_session, "sub123", [ext])

            assert len(results) == 1
            mock_embed.assert_called_once()
            call_texts = mock_embed.call_args[0][0]
            assert "Daft Punk - Discovery" in call_texts
            assert "Daft Punk" in call_texts

            # Check executed SQL statements
            executed_stmts = [call.args[0] for call in mock_session.execute.call_args_list]

            # Find album, artist, and release insert statements
            album_insert = None
            art_insert = None
            rel_insert = None
            for stmt in executed_stmts:
                if hasattr(stmt, "table"):
                    if stmt.table.name == "albums":
                        album_insert = stmt
                    elif stmt.table.name == "artists":
                        art_insert = stmt
                    elif stmt.table.name == "releases":
                        rel_insert = stmt

            assert album_insert is not None, "Album insert statement must be executed"
            assert art_insert is not None, "Artist insert statement must be executed"
            assert rel_insert is not None, "Release insert statement must be executed"

            compiled_album = album_insert.compile(dialect=postgresql.dialect())
            assert "embedding" in compiled_album.params
            assert compiled_album.params["embedding"] == dummy_vec_album
            assert compiled_album.params["discogs_status"] == "pending"

            compiled_art = art_insert.compile(dialect=postgresql.dialect())
            assert "embedding" in compiled_art.params
            assert compiled_art.params["embedding"] == dummy_vec_art

            compiled_rel = rel_insert.compile(dialect=postgresql.dialect())
            assert compiled_rel.params["album_id"] == "album-uuid-1"

    def test_persist_multi_artist_generates_and_stores_all_artist_embeddings(self):
        mock_session = MagicMock()
        mock_sub = MagicMock(spec=Submission)
        mock_sub.id = "sub456"

        mock_release = MagicMock(spec=Release)
        mock_release.id = "rel-uuid-2"

        def get_side_effect(model_cls, pk):
            if model_cls is Submission:
                return mock_sub
            if model_cls is Release:
                return mock_release
            return None

        mock_session.get.side_effect = get_side_effect
        mock_session.execute.return_value.scalar_one.side_effect = [
            "album-uuid-split",
            "art-uuid-1",
            "art-uuid-2",
            "rel-uuid-2",
        ]

        ext = ExtractedRelease(
            artists=["Converge", "Dropdead"],
            combined_artist="Converge / Dropdead",
            album_title="Split",
            variant=None,
            retailer=None,
        )

        with patch("reddit_scrape.persistence.embed_texts") as mock_embed:
            mock_embed.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

            results = persist_extracted_releases(mock_session, "sub456", [ext])
            assert len(results) == 1

            call_texts = mock_embed.call_args[0][0]
            assert "Converge / Dropdead - Split" in call_texts
            assert "Converge" in call_texts
            assert "Dropdead" in call_texts

    def test_release_and_artist_on_conflict_updates_embedding(self):
        mock_session = MagicMock()
        mock_sub = MagicMock(spec=Submission)
        mock_sub.id = "sub789"
        mock_session.get.return_value = mock_sub

        mock_session.execute.return_value.scalar_one.side_effect = [
            "album-uuid-3",
            "art-uuid-3",
            "rel-uuid-3",
        ]

        ext = ExtractedRelease(
            artists=["Artist"],
            combined_artist="Artist",
            album_title="Album",
            variant=None,
            retailer=None,
        )

        with patch("reddit_scrape.persistence.embed_texts") as mock_embed:
            mock_embed.return_value = [[0.1] * 384, [0.2] * 384]
            persist_extracted_releases(mock_session, "sub789", [ext])

            executed_stmts = [call.args[0] for call in mock_session.execute.call_args_list]
            album_insert = next(s for s in executed_stmts if hasattr(s, "table") and s.table.name == "albums")
            art_insert = next(s for s in executed_stmts if hasattr(s, "table") and s.table.name == "artists")

            # Verify on conflict clauses contain embedding update/coalesce
            compiled_album = str(album_insert.compile(dialect=postgresql.dialect()))
            assert "embedding" in compiled_album
            assert "ON CONFLICT (fingerprint) DO UPDATE" in compiled_album

            compiled_art = str(art_insert.compile(dialect=postgresql.dialect()))
            assert "embedding" in compiled_art
            assert "ON CONFLICT (name) DO UPDATE" in compiled_art


class TestCatalogCandidateRetrievalUnit:
    def test_empty_or_whitespace_queries_return_empty_candidates(self):
        mock_session = MagicMock()
        with patch("reddit_scrape.persistence.embed_texts") as mock_embed:
            candidates_empty = find_catalog_candidates(mock_session, [])
            assert candidates_empty.is_empty
            assert not candidates_empty
            assert candidates_empty.releases == []
            assert candidates_empty.artists == []

            candidates_whitespace = find_catalog_candidates(mock_session, ["   ", "\t\n  "])
            assert candidates_whitespace.is_empty
            assert not candidates_whitespace

            mock_embed.assert_not_called()
            mock_session.execute.assert_not_called()

    def test_single_batch_embedding_called_with_pre_cleaned_titles(self):
        mock_session = MagicMock()
        # Mock execute returning empty query results
        mock_session.execute.return_value.all.return_value = []

        titles = [
            "  Converge \u2014 Dropdead \u2013 Split   ",
            "Daft   Punk  - Discovery  ",
            "  Converge \u2014 Dropdead \u2013 Split   ",  # duplicate title in batch
        ]

        dummy_vector = [0.05] * 384
        with patch("reddit_scrape.persistence.embed_texts") as mock_embed:
            mock_embed.return_value = [dummy_vector, dummy_vector]

            find_catalog_candidates(mock_session, titles, top_k=5, min_similarity=0.50)

            mock_embed.assert_called_once()
            embedded_titles = mock_embed.call_args[0][0]
            # Must be pre-cleaned and deduplicated for embedding
            assert embedded_titles == [
                "Converge - Dropdead - Split",
                "Daft Punk - Discovery",
            ]

    def test_similarity_queries_filter_by_max_distance_and_limit_top_k(self):
        mock_session = MagicMock()
        mock_session.execute.return_value.all.return_value = []

        dummy_vector = [0.1] * 384
        with patch("reddit_scrape.persistence.embed_texts") as mock_embed:
            mock_embed.return_value = [dummy_vector]

            find_catalog_candidates(
                mock_session,
                "Converge - Jane Doe",
                top_k=3,
                min_similarity=0.60,
            )

            # Executed queries should include Album and Artist distance queries
            executed_stmts = [call.args[0] for call in mock_session.execute.call_args_list]
            assert len(executed_stmts) >= 2

            # Inspect album query
            album_stmt = executed_stmts[0]
            compiled_album = str(album_stmt.compile(dialect=postgresql.dialect()))
            assert "FROM albums" in compiled_album
            assert "albums.embedding <=>" in compiled_album
            assert "ORDER BY" in compiled_album
            assert "LIMIT" in compiled_album

            # Inspect artist query
            art_stmt = executed_stmts[1]
            compiled_art = str(art_stmt.compile(dialect=postgresql.dialect()))
            assert "FROM artists" in compiled_art
            assert "artists.embedding <=>" in compiled_art
            assert "ORDER BY" in compiled_art
            assert "LIMIT" in compiled_art

    def test_contributing_artists_hydrated_from_release_artists(self):
        mock_session = MagicMock()

        album_mock = MagicMock(spec=Album)
        album_mock.id = "album-uuid-split"
        album_mock.combined_artist = "Converge / Dropdead"
        album_mock.title = "Split"

        rel_mock = MagicMock(spec=Release)
        rel_mock.id = "rel-uuid-split"
        rel_mock.album_id = "album-uuid-split"
        rel_mock.variant = "Clear 7\""
        rel_mock.retailer_name = "Deathwish"

        art_converge = MagicMock(spec=Artist)
        art_converge.name = "Converge"

        # Side effects for session.execute(...).all() and .scalars().all()
        # 1. Album query -> returns [(album_mock, 0.05)] (sim = 0.95)
        # 2. Artist query -> returns [(art_converge, 0.10)] (sim = 0.90)
        # 3. AlbumArtist hydration query -> returns [("album-uuid-split", "Converge"), ("album-uuid-split", "Dropdead")]
        # 4. Release hydration query -> returns [rel_mock]
        mock_session.execute.return_value.all.side_effect = [
            [(album_mock, 0.05)],
            [(art_converge, 0.10)],
            [("album-uuid-split", "Converge"), ("album-uuid-split", "Dropdead")],
        ]
        mock_session.execute.return_value.scalars.return_value.all.return_value = [rel_mock]

        dummy_vector = [0.1] * 384
        with patch("reddit_scrape.persistence.embed_texts") as mock_embed:
            mock_embed.return_value = [dummy_vector]

            result = find_catalog_candidates(
                mock_session,
                "Dropdead / Converge - Split 7\"",
                top_k=5,
                min_similarity=0.50,
            )

            assert len(result.releases) == 1
            cand = result.releases[0]
            assert cand.id == "rel-uuid-split"
            assert cand.album_id == "album-uuid-split"
            assert cand.combined_artist == "Converge / Dropdead"
            assert cand.album_title == "Split"
            assert cand.variant == "Clear 7\""
            assert cand.retailer == "Deathwish"
            assert cand.artists == ["Converge", "Dropdead"]
            assert cand.similarity == 0.95

            assert result.artists == ["Converge"]

    def test_batch_deduplication_keeps_highest_similarity(self):
        mock_session = MagicMock()

        album_mock = MagicMock(spec=Album)
        album_mock.id = "album-uuid-common"
        album_mock.combined_artist = "Daft Punk"
        album_mock.title = "Discovery"

        art_mock = MagicMock(spec=Artist)
        art_mock.name = "Daft Punk"

        rel_mock = MagicMock(spec=Release)
        rel_mock.id = "rel-uuid-common"
        rel_mock.album_id = "album-uuid-common"
        rel_mock.variant = None
        rel_mock.retailer_name = None

        # 2 queries in batch
        # Query 1: album with dist 0.20 (sim 0.80), art with dist 0.25 (sim 0.75)
        # Query 2: same album with dist 0.05 (sim 0.95), same art with dist 0.02 (sim 0.98)
        # Hydration queries: [("album-uuid-common", "Daft Punk")]
        mock_session.execute.return_value.all.side_effect = [
            [(album_mock, 0.20)],
            [(art_mock, 0.25)],
            [(album_mock, 0.05)],
            [(art_mock, 0.02)],
            [("album-uuid-common", "Daft Punk")],
        ]
        mock_session.execute.return_value.scalars.return_value.all.return_value = [rel_mock]

        dummy_vector = [0.1] * 384
        with patch("reddit_scrape.persistence.embed_texts") as mock_embed:
            mock_embed.return_value = [dummy_vector, dummy_vector]

            result = find_catalog_candidates(
                mock_session,
                ["Discovery - Daft Punk", "Daft Punk - Discovery"],
                top_k=5,
                min_similarity=0.50,
            )

            # Deduplicated: only 1 release and 1 artist returned
            assert len(result.releases) == 1
            assert result.releases[0].id == "rel-uuid-common"
            assert result.releases[0].similarity == 0.95

            assert len(result.artists) == 1
            assert result.artists == ["Daft Punk"]

    def test_candidates_below_min_similarity_excluded(self):
        mock_session = MagicMock()

        album_mock = MagicMock(spec=Album)
        album_mock.id = "album-uuid-low"
        album_mock.combined_artist = "Low Sim"
        album_mock.title = "Album"

        art_mock = MagicMock(spec=Artist)
        art_mock.name = "Low Sim Artist"

        # Distance 0.55 -> similarity 0.45, which is < min_similarity 0.50
        mock_session.execute.return_value.all.side_effect = [
            [(album_mock, 0.55)],
            [(art_mock, 0.55)],
        ]

        dummy_vector = [0.1] * 384
        with patch("reddit_scrape.persistence.embed_texts") as mock_embed:
            mock_embed.return_value = [dummy_vector]

            result = find_catalog_candidates(
                mock_session,
                "Completely Unrelated Query",
                top_k=5,
                min_similarity=0.50,
            )

            assert len(result.releases) == 0
            assert len(result.artists) == 0
            assert result.is_empty

    def test_service_find_candidates_manages_session(self):
        mock_session_factory = MagicMock()
        mock_session = MagicMock()
        mock_session_factory.return_value.__enter__.return_value = mock_session

        service = ReleasePersistenceService(mock_session_factory)

        with patch("reddit_scrape.persistence.find_catalog_candidates") as mock_find:
            mock_find.return_value = CatalogCandidates()

            result = service.find_candidates("Daft Punk - Discovery", top_k=3, min_similarity=0.7)

            mock_session_factory.assert_called_once()
            mock_find.assert_called_once_with(
                session=mock_session,
                query_titles="Daft Punk - Discovery",
                top_k=3,
                min_similarity=0.7,
            )
            assert result.is_empty

    def test_catalog_candidates_dataclass_properties(self):
        empty = CatalogCandidates()
        assert empty.is_empty
        assert not empty

        cand = CatalogReleaseCandidate(
            id="rel-123",
            combined_artist="Converge / Dropdead",
            album_title="Split",
            artists=["Converge", "Dropdead"],
            variant="Clear 7\"",
            retailer="Deathwish",
            similarity=0.9521,
        )
        assert cand.id == "rel-123"
        assert cand.combined_artist == "Converge / Dropdead"
        assert cand.artists == ["Converge", "Dropdead"]
        assert cand.retailer == "Deathwish"
        assert cand.similarity == 0.9521

        non_empty = CatalogCandidates(releases=[cand], artists=["Converge"])
        assert not non_empty.is_empty
        assert bool(non_empty)
