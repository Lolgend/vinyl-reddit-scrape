from unittest.mock import MagicMock, patch

from pgvector.sqlalchemy import Vector

from reddit_scrape.db import Base, init_db, reset_db
from reddit_scrape.models import Album, AlbumArtist, Artist, Release, SubmissionRelease


class TestDatabaseSchemaAndInit:
    def test_artist_model_columns_and_hnsw_index(self):
        col_emb = Artist.__table__.columns.get("embedding")
        assert col_emb is not None, "Artist must have an embedding column"
        assert isinstance(col_emb.type, Vector)
        assert col_emb.type.dim == 384
        assert col_emb.nullable is True

        col_discogs = Artist.__table__.columns.get("discogs_artist_id")
        assert col_discogs is not None, "Artist must have a discogs_artist_id column"
        assert col_discogs.nullable is True

        col_updated = Artist.__table__.columns.get("updated_at")
        assert col_updated is not None, "Artist must have an updated_at column"

        indexes = {idx.name: idx for idx in Artist.__table__.indexes}
        assert "idx_artists_embedding" in indexes, "idx_artists_embedding index must exist on Artist"
        idx = indexes["idx_artists_embedding"]
        assert idx.dialect_options["postgresql"]["using"] == "hnsw"
        assert idx.dialect_options["postgresql"]["ops"] == {"embedding": "vector_cosine_ops"}

        assert "idx_artists_discogs_artist_id" in indexes or col_discogs.unique is True

    def test_album_model_columns_and_indexes(self):
        table = Album.__table__
        assert table.name == "albums"

        col_artist = table.columns.get("combined_artist")
        assert col_artist is not None and not col_artist.nullable

        col_title = table.columns.get("title")
        assert col_title is not None and not col_title.nullable

        col_fp = table.columns.get("fingerprint")
        assert col_fp is not None and not col_fp.nullable and col_fp.unique

        col_master = table.columns.get("discogs_master_id")
        assert col_master is not None and col_master.nullable and col_master.unique

        col_release = table.columns.get("discogs_release_id")
        assert col_release is not None and col_release.nullable and col_release.unique

        col_status = table.columns.get("discogs_status")
        assert col_status is not None and not col_status.nullable

        col_attempts = table.columns.get("enrichment_attempts")
        assert col_attempts is not None and not col_attempts.nullable

        col_emb = table.columns.get("embedding")
        assert col_emb is not None
        assert isinstance(col_emb.type, Vector)
        assert col_emb.type.dim == 384
        assert col_emb.nullable is True

        indexes = {idx.name: idx for idx in table.indexes}
        assert "idx_albums_embedding" in indexes, "idx_albums_embedding index must exist on Album"
        idx_emb = indexes["idx_albums_embedding"]
        assert idx_emb.dialect_options["postgresql"]["using"] == "hnsw"
        assert idx_emb.dialect_options["postgresql"]["ops"] == {"embedding": "vector_cosine_ops"}

        assert "idx_albums_enrichment_queue" in indexes, "idx_albums_enrichment_queue must exist on Album"
        idx_queue = indexes["idx_albums_enrichment_queue"]
        queue_cols = [c.name for c in idx_queue.columns]
        assert queue_cols == ["discogs_status", "last_enriched_at", "created_at"]
        assert idx_queue.dialect_options["postgresql"]["where"] is not None

    def test_album_artists_model_structure(self):
        table = AlbumArtist.__table__
        assert table.name == "album_artists"

        pk_cols = [c.name for c in table.primary_key.columns]
        assert set(pk_cols) == {"album_id", "artist_id"}

        fks = {fk.target_fullname for fk in table.foreign_keys}
        assert "albums.id" in fks
        assert "artists.id" in fks

    def test_release_model_columns_and_indexes(self):
        table = Release.__table__
        assert table.name == "releases"

        col_album_id = table.columns.get("album_id")
        assert col_album_id is not None and not col_album_id.nullable

        fks = {fk.target_fullname for fk in table.foreign_keys}
        assert "albums.id" in fks

        col_fp = table.columns.get("fingerprint")
        assert col_fp is not None and not col_fp.nullable and col_fp.unique

        # Release must NOT have embedding or combined_artist or album_title in 3-tier layout
        assert "embedding" not in table.columns
        assert "combined_artist" not in table.columns
        assert "album_title" not in table.columns

    def test_submission_release_model_structure(self):
        table = SubmissionRelease.__table__
        assert table.name == "submission_releases"

        pk_cols = [c.name for c in table.primary_key.columns]
        assert set(pk_cols) == {"submission_id", "release_id"}

    def test_init_db_executes_create_extension_vector(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        create_all_mock = MagicMock()
        drop_all_mock = MagicMock()
        monkeypatch.setattr(Base.metadata, "create_all", create_all_mock)
        monkeypatch.setattr(Base.metadata, "drop_all", drop_all_mock)

        with patch("reddit_scrape.db.inspect") as mock_inspect:
            mock_inspector = MagicMock()
            mock_inspector.get_table_names.return_value = ["submissions"]
            mock_inspect.return_value = mock_inspector

            init_db(mock_engine)

            executed_sqls = [str(call.args[0]) for call in mock_conn.execute.call_args_list]
            assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in executed_sqls)
            create_all_mock.assert_called_once_with(mock_engine)
            drop_all_mock.assert_not_called()

    def test_init_db_drops_incompatible_legacy_tables(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        create_all_mock = MagicMock()
        monkeypatch.setattr(Base.metadata, "create_all", create_all_mock)

        with patch("reddit_scrape.db.inspect") as mock_inspect:
            mock_inspector = MagicMock()
            # Legacy 2-tier database has release_artists table
            mock_inspector.get_table_names.return_value = [
                "submissions",
                "releases",
                "release_artists",
                "artists",
            ]
            mock_inspector.get_columns.side_effect = lambda tname: (
                [{"name": "combined_artist"}] if tname == "releases" else [{"name": "name"}]
            )
            mock_inspect.return_value = mock_inspector

            init_db(mock_engine)

            executed_sqls = [str(call.args[0]) for call in mock_conn.execute.call_args_list]
            # Must drop legacy tables
            assert any("DROP TABLE IF EXISTS" in s and "release_artists" in s for s in executed_sqls)
            assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in executed_sqls)
            create_all_mock.assert_called_once_with(mock_engine)

    def test_init_db_with_drop_existing_drops_before_create(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        create_all_mock = MagicMock()
        drop_all_mock = MagicMock()
        monkeypatch.setattr(Base.metadata, "create_all", create_all_mock)
        monkeypatch.setattr(Base.metadata, "drop_all", drop_all_mock)

        with patch("reddit_scrape.db.inspect") as mock_inspect:
            mock_inspector = MagicMock()
            mock_inspector.get_table_names.return_value = []
            mock_inspect.return_value = mock_inspector

            init_db(mock_engine, drop_existing=True)

            drop_all_mock.assert_called_once_with(mock_engine)
            create_all_mock.assert_called_once_with(mock_engine)

    def test_reset_db_drops_and_recreates_tables(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        create_all_mock = MagicMock()
        drop_all_mock = MagicMock()
        monkeypatch.setattr(Base.metadata, "create_all", create_all_mock)
        monkeypatch.setattr(Base.metadata, "drop_all", drop_all_mock)

        with patch("reddit_scrape.db.inspect") as mock_inspect:
            mock_inspector = MagicMock()
            mock_inspector.get_table_names.return_value = []
            mock_inspect.return_value = mock_inspector

            reset_db(mock_engine)

            drop_all_mock.assert_called_once_with(mock_engine)
            create_all_mock.assert_called_once_with(mock_engine)
