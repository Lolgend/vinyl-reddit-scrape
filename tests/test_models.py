from datetime import datetime, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.models import (
    Album,
    AlbumArtist,
    Artist,
    Release,
    Submission,
    SubmissionRelease,
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


class TestModelsIntegration:
    def test_create_album_artist_release_hierarchy(self, session_factory):
        saved_album_id = None
        saved_art1_id = None
        saved_art2_id = None
        saved_rel1_id = None

        with session_factory() as session, session.begin():
            # 1. Constituent artists
            art1 = Artist(name="Converge", discogs_artist_id=262795)
            art2 = Artist(name="Dropdead", discogs_artist_id=304415)
            session.add_all([art1, art2])
            session.flush()

            # 2. Canonical Album
            album = Album(
                combined_artist="Converge / Dropdead",
                title="Split",
                fingerprint="test_album_fp_001",
                discogs_master_id=12345,
                discogs_status="pending",
                embedding=[0.01] * 384,
            )
            session.add(album)
            session.flush()

            # 3. Link album artists
            aa1 = AlbumArtist(album_id=album.id, artist_id=art1.id)
            aa2 = AlbumArtist(album_id=album.id, artist_id=art2.id)
            session.add_all([aa1, aa2])

            # 4. Pressing / Release
            rel1 = Release(
                album_id=album.id,
                variant="Clear 7\"",
                retailer_name="Deathwish",
                fingerprint="test_rel_fp_001",
            )
            rel2 = Release(
                album_id=album.id,
                variant="Black 7\"",
                retailer_name=None,
                fingerprint="test_rel_fp_002",
            )
            session.add_all([rel1, rel2])

            # 5. Submission
            sub = Submission(
                id="sub_test_mod_1",
                reddit_id="mod1",
                title="Converge / Dropdead - Split",
                url="https://reddit.com/mod1",
                created_utc=datetime.now(timezone.utc),
            )
            session.add(sub)
            session.flush()

            # 6. Submission release link
            sr = SubmissionRelease(submission_id=sub.id, release_id=rel1.id)
            session.add(sr)

            saved_album_id = album.id
            saved_art1_id = art1.id
            saved_art2_id = art2.id
            saved_rel1_id = rel1.id

        # Verify query and cascading relations
        with session_factory() as session:
            retrieved_album = session.get(Album, saved_album_id)
            assert retrieved_album is not None
            assert retrieved_album.combined_artist == "Converge / Dropdead"
            assert retrieved_album.discogs_status == "pending"
            assert retrieved_album.enrichment_attempts == 0
            assert retrieved_album.embedding is not None
            assert len(retrieved_album.embedding) == 384

            # Verify album artists
            links = session.execute(
                select(AlbumArtist).where(AlbumArtist.album_id == saved_album_id)
            ).scalars().all()
            assert len(links) == 2
            linked_artist_ids = {l.artist_id for l in links}
            assert linked_artist_ids == {saved_art1_id, saved_art2_id}

            # Verify child releases
            releases = session.execute(
                select(Release).where(Release.album_id == saved_album_id)
            ).scalars().all()
            assert len(releases) == 2
            variants = {r.variant for r in releases}
            assert variants == {"Clear 7\"", "Black 7\""}

            # Verify release property accessors
            rel = session.get(Release, saved_rel1_id)
            assert rel is not None
            assert rel.combined_artist == "Converge / Dropdead"
            assert rel.album_title == "Split"
