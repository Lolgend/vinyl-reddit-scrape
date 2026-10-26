from unittest.mock import MagicMock, patch

from pgvector.sqlalchemy import Vector

from reddit_scrape.db import Base, init_db, reset_db
from reddit_scrape.models import Artist, Release


class TestDatabaseSchemaAndInit:
    def test_artist_model_embedding_column_and_hnsw_index(self):
        col = Artist.__table__.columns.get("embedding")
        assert col is not None, "Artist must have an embedding column"
        assert isinstance(col.type, Vector)
        assert col.type.dim == 384
        assert col.nullable is True

        indexes = {idx.name: idx for idx in Artist.__table__.indexes}
        assert "idx_artists_embedding" in indexes, "idx_artists_embedding index must exist on Artist"
        idx = indexes["idx_artists_embedding"]
        assert idx.dialect_options["postgresql"]["using"] == "hnsw"
        assert idx.dialect_options["postgresql"]["ops"] == {"embedding": "vector_cosine_ops"}

    def test_release_model_embedding_column_and_hnsw_index(self):
        col = Release.__table__.columns.get("embedding")
        assert col is not None, "Release must have an embedding column"
        assert isinstance(col.type, Vector)
        assert col.type.dim == 384
        assert col.nullable is True

        indexes = {idx.name: idx for idx in Release.__table__.indexes}
        assert "idx_releases_embedding" in indexes, "idx_releases_embedding index must exist on Release"
        idx = indexes["idx_releases_embedding"]
        assert idx.dialect_options["postgresql"]["using"] == "hnsw"
        assert idx.dialect_options["postgresql"]["ops"] == {"embedding": "vector_cosine_ops"}

    def test_init_db_executes_create_extension_vector(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        create_all_mock = MagicMock()
        drop_all_mock = MagicMock()
        monkeypatch.setattr(Base.metadata, "create_all", create_all_mock)
        monkeypatch.setattr(Base.metadata, "drop_all", drop_all_mock)

        init_db(mock_engine)

        # Verify extension execution occurred
        executed_sqls = [str(call.args[0]) for call in mock_conn.execute.call_args_list]
        assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in executed_sqls)
        create_all_mock.assert_called_once_with(mock_engine)
        drop_all_mock.assert_not_called()

    def test_init_db_with_drop_existing_drops_before_create(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        create_all_mock = MagicMock()
        drop_all_mock = MagicMock()
        monkeypatch.setattr(Base.metadata, "create_all", create_all_mock)
        monkeypatch.setattr(Base.metadata, "drop_all", drop_all_mock)

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

        reset_db(mock_engine)

        drop_all_mock.assert_called_once_with(mock_engine)
        executed_sqls = [str(call.args[0]) for call in mock_conn.execute.call_args_list]
        assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in executed_sqls)
        create_all_mock.assert_called_once_with(mock_engine)

    def test_reset_db_uses_default_engine_when_none(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__.return_value = mock_conn

        with patch("reddit_scrape.db.get_engine", return_value=mock_engine) as mock_get_engine:
            create_all_mock = MagicMock()
            drop_all_mock = MagicMock()
            monkeypatch.setattr(Base.metadata, "create_all", create_all_mock)
            monkeypatch.setattr(Base.metadata, "drop_all", drop_all_mock)

            reset_db()

            mock_get_engine.assert_called_once()
            drop_all_mock.assert_called_once_with(mock_engine)
            create_all_mock.assert_called_once_with(mock_engine)
