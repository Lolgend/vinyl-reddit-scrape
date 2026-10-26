from unittest.mock import MagicMock, patch

from sqlalchemy.dialects import postgresql

from reddit_scrape.extraction import ExtractedRelease
from reddit_scrape.models import Release, Submission
from reddit_scrape.persistence import persist_extracted_releases


class TestPersistenceEmbeddingsUnit:
    def test_persist_single_artist_generates_and_stores_embeddings(self):
        mock_session = MagicMock()
        mock_sub = MagicMock(spec=Submission)
        mock_sub.id = "sub123"
        mock_session.get.return_value = mock_sub

        mock_release = MagicMock(spec=Release)
        mock_release.id = "rel-uuid-1"

        # Mock execute returning scalar_one for release and artist inserts
        mock_session.execute.return_value.scalar_one.side_effect = ["rel-uuid-1", "art-uuid-1"]

        # When session.get(Release, ...) is called at end of loop
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

        dummy_vec_rel = [0.1] * 384
        dummy_vec_art = [0.2] * 384

        def fake_embed(texts):
            result = []
            for t in texts:
                if "Discovery" in t:
                    result.append(dummy_vec_rel)
                else:
                    result.append(dummy_vec_art)
            return result

        with patch("reddit_scrape.persistence.embed_texts") as mock_embed:
            mock_embed.side_effect = fake_embed

            results = persist_extracted_releases(mock_session, "sub123", [ext])

            assert len(results) == 1
            mock_embed.assert_called_once()
            call_texts = mock_embed.call_args[0][0]
            assert "Daft Punk - Discovery (2LP Gatefold) [Rough Trade]" in call_texts
            assert "Daft Punk" in call_texts

            # Check executed SQL statements
            executed_stmts = [call.args[0] for call in mock_session.execute.call_args_list]

            # Find release insert statement
            rel_insert = None
            art_insert = None
            for stmt in executed_stmts:
                if hasattr(stmt, "table"):
                    if stmt.table.name == "releases":
                        rel_insert = stmt
                    elif stmt.table.name == "artists":
                        art_insert = stmt

            assert rel_insert is not None, "Release insert statement must be executed"
            assert art_insert is not None, "Artist insert statement must be executed"

            compiled_rel = rel_insert.compile(dialect=postgresql.dialect())
            assert "embedding" in compiled_rel.params
            assert compiled_rel.params["embedding"] == dummy_vec_rel

            compiled_art = art_insert.compile(dialect=postgresql.dialect())
            assert "embedding" in compiled_art.params
            assert compiled_art.params["embedding"] == dummy_vec_art

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
            "rel-uuid-2",
            "art-uuid-1",
            "art-uuid-2",
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

        mock_session.execute.return_value.scalar_one.side_effect = ["rel-uuid-3", "art-uuid-3"]

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
            rel_insert = next(s for s in executed_stmts if hasattr(s, "table") and s.table.name == "releases")
            art_insert = next(s for s in executed_stmts if hasattr(s, "table") and s.table.name == "artists")

            # Verify on conflict clauses contain embedding update/coalesce
            compiled_rel = str(rel_insert.compile(dialect=postgresql.dialect()))
            assert "embedding" in compiled_rel
            assert "ON CONFLICT (fingerprint) DO UPDATE" in compiled_rel

            compiled_art = str(art_insert.compile(dialect=postgresql.dialect()))
            assert "embedding" in compiled_art
            assert "ON CONFLICT (name) DO UPDATE" in compiled_art
