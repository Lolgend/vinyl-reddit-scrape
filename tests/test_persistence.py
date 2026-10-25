import os
from datetime import datetime, timezone
import pytest
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.models import (
    Artist,
    Release,
    ReleaseArtist,
    Submission,
    SubmissionRelease,
)
from reddit_scrape.extraction import ExtractedRelease, ExtractionResult
from reddit_scrape.persistence import (
    STANDARD_TOKEN,
    ReleasePersistenceService,
    compute_release_fingerprint,
    normalize_attribute,
    persist_extracted_releases,
)


@pytest.fixture(scope="session")
def engine():
    try:
        eng = get_engine()
        init_db(eng)
        return eng
    except Exception as exc:
        pytest.skip(f"Database unavailable: {exc}")


@pytest.fixture(autouse=True)
def clean_db(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE submission_snapshots, submission_releases, release_artists, releases, artists, submissions CASCADE;"
            )
        )
    yield
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE submission_snapshots, submission_releases, release_artists, releases, artists, submissions CASCADE;"
            )
        )


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine)


def create_test_submission(session, sub_id: str, reddit_id: str, title: str) -> Submission:
    sub = Submission(
        id=sub_id,
        reddit_id=reddit_id,
        title=title,
        url=f"https://reddit.com/{reddit_id}",
        permalink=f"/r/vinylreleases/comments/{reddit_id}",
        flair="New Release",
        created_utc=datetime.now(timezone.utc),
        is_extracted=False,
        is_release=True,
    )
    session.add(sub)
    session.flush()
    return sub


class TestReleaseFingerprint:
    def test_normalization_and_standard_token(self):
        """Verify lowercase, collapsed whitespace, normalized dashes, and [standard] placeholder."""
        fp1 = compute_release_fingerprint(
            combined_artist="  Daft   Punk  ",
            album_title="Discovery",
            variant=None,
            retailer=None,
        )
        fp2 = compute_release_fingerprint(
            combined_artist="daft punk",
            album_title="discovery",
            variant="",
            retailer="   ",
        )
        assert fp1 == fp2
        assert len(fp1) == 64

    def test_punctuation_preservation(self):
        """Punctuation like '/', '&', and commas are preserved in normalized attributes and fingerprints."""
        assert normalize_attribute("Pink /500") == "pink /500"
        assert normalize_attribute("Converge / Dropdead") == "converge / dropdead"
        assert normalize_attribute("Tyler, The Creator") == "tyler, the creator"

        fp1 = compute_release_fingerprint(
            combined_artist="Tyler, The Creator",
            album_title="IGOR",
            variant="Pink /500",
            retailer="Golf Wang",
        )
        fp2 = compute_release_fingerprint(
            combined_artist="Tyler The Creator",
            album_title="IGOR",
            variant="Pink 500",
            retailer="Golf Wang",
        )
        assert fp1 != fp2

    def test_unicode_dash_normalization(self):
        """Unicode dashes (\u2013, \u2014, etc.) normalize to hyphen."""
        fp1 = compute_release_fingerprint(
            combined_artist="Artist",
            album_title="Album \u2013 Part 1",
            variant="180g \u2014 Limited",
            retailer="Store",
        )
        fp2 = compute_release_fingerprint(
            combined_artist="Artist",
            album_title="Album - Part 1",
            variant="180g - Limited",
            retailer="Store",
        )
        assert fp1 == fp2

    def test_collision_prevention_standard_vs_exclusive(self):
        """Standard pressings must never collide with variant pressings or retailer exclusives."""
        fp_standard = compute_release_fingerprint(
            combined_artist="Radiohead",
            album_title="OK Computer",
            variant=None,
            retailer=None,
        )
        fp_variant = compute_release_fingerprint(
            combined_artist="Radiohead",
            album_title="OK Computer",
            variant="Blue Opaque Vinyl",
            retailer=None,
        )
        fp_retailer = compute_release_fingerprint(
            combined_artist="Radiohead",
            album_title="OK Computer",
            variant=None,
            retailer="Rough Trade",
        )
        assert fp_standard != fp_variant
        assert fp_standard != fp_retailer
        assert fp_variant != fp_retailer


class TestReleasePersistence:
    def test_persist_single_artist_release(self, session_factory):
        """Verify single-artist release persistence, artist row, and join records."""
        with session_factory() as session:
            with session.begin():
                sub = create_test_submission(session, "sub1", "red1", "Daft Punk - Discovery")

        service = ReleasePersistenceService(session_factory)
        ext_release = ExtractedRelease(
            artists=["Daft Punk"],
            combined_artist="Daft Punk",
            album_title="Discovery",
            variant="2LP Gatefold",
            retailer=None,
        )
        result = service.persist("sub1", [ext_release])

        assert len(result) == 1
        rel = result[0]
        assert rel.combined_artist == "Daft Punk"
        assert rel.album_title == "Discovery"
        assert rel.variant == "2LP Gatefold"
        assert rel.retailer_name is None
        assert rel.fingerprint is not None

        with session_factory() as session:
            # Check submission updated
            updated_sub = session.get(Submission, "sub1")
            assert updated_sub.is_extracted is True

            # Check artist created
            artists = session.execute(select(Artist)).scalars().all()
            assert len(artists) == 1
            assert artists[0].name == "Daft Punk"

            # Check release_artists link
            rel_artists = session.execute(select(ReleaseArtist)).scalars().all()
            assert len(rel_artists) == 1
            assert rel_artists[0].release_id == rel.id
            assert rel_artists[0].artist_id == artists[0].id

            # Check submission_releases link
            sub_releases = session.execute(select(SubmissionRelease)).scalars().all()
            assert len(sub_releases) == 1
            assert sub_releases[0].submission_id == "sub1"
            assert sub_releases[0].release_id == rel.id

    def test_persist_multi_artist_release(self, session_factory):
        """Verify multi-artist split/collaboration associates each artist individually."""
        with session_factory() as session:
            with session.begin():
                sub = create_test_submission(session, "sub2", "red2", "Converge / Dropdead Split")

        service = ReleasePersistenceService(session_factory)
        ext_release = ExtractedRelease(
            artists=["Converge", "Dropdead"],
            combined_artist="Converge / Dropdead",
            album_title="Converge / Dropdead Split",
            variant="Clear /500",
            retailer="Deathwish",
        )
        result = service.persist("sub2", [ext_release])

        assert len(result) == 1
        rel = result[0]

        with session_factory() as session:
            artists = session.execute(select(Artist).order_by(Artist.name)).scalars().all()
            assert len(artists) == 2
            artist_names = [a.name for a in artists]
            assert "Converge" in artist_names
            assert "Dropdead" in artist_names

            # Check both artists linked to this release
            links = session.execute(
                select(ReleaseArtist).where(ReleaseArtist.release_id == rel.id)
            ).scalars().all()
            assert len(links) == 2

    def test_deduplication_two_submissions_same_release(self, session_factory):
        """Two different submissions for the exact same pressing resolve to the same Release row."""
        with session_factory() as session:
            with session.begin():
                create_test_submission(session, "sub_a", "red_a", "Post 1: Charli XCX - Brat Pink Vinyl")
                create_test_submission(session, "sub_b", "red_b", "Post 2: Charli XCX - Brat [Pink Vinyl]")

        service = ReleasePersistenceService(session_factory)
        ext1 = ExtractedRelease(
            artists=["Charli XCX"],
            combined_artist="Charli XCX",
            album_title="Brat",
            variant="Pink Vinyl",
            retailer="Rough Trade",
        )
        ext2 = ExtractedRelease(
            artists=["Charli XCX"],
            combined_artist="Charli XCX",
            album_title="Brat",
            variant="Pink Vinyl",
            retailer="Rough Trade",
        )

        res1 = service.persist("sub_a", [ext1])
        res2 = service.persist("sub_b", [ext2])

        assert len(res1) == 1
        assert len(res2) == 1
        # Both must point to the identical Release ID
        assert res1[0].id == res2[0].id

        with session_factory() as session:
            # Only 1 canonical Release record
            all_releases = session.execute(select(Release)).scalars().all()
            assert len(all_releases) == 1

            # Only 1 canonical Artist record
            all_artists = session.execute(select(Artist)).scalars().all()
            assert len(all_artists) == 1

            # 2 rows in submission_releases (one per submission)
            sub_rels = session.execute(select(SubmissionRelease)).scalars().all()
            assert len(sub_rels) == 2
            linked_sub_ids = {sr.submission_id for sr in sub_rels}
            assert linked_sub_ids == {"sub_a", "sub_b"}

            # Both submissions marked is_extracted = True
            sub_a = session.get(Submission, "sub_a")
            sub_b = session.get(Submission, "sub_b")
            assert sub_a.is_extracted is True
            assert sub_b.is_extracted is True

    def test_empty_release_handling(self, session_factory):
        """When releases: [] (e.g. non-release or generic restock), mark is_extracted=True without creating releases."""
        with session_factory() as session:
            with session.begin():
                create_test_submission(session, "sub_disc", "red_disc", "[Discussion] Favorite turntables?")

        service = ReleasePersistenceService(session_factory)
        result = service.persist("sub_disc", ExtractionResult(releases=[]))

        assert result == []

        with session_factory() as session:
            sub = session.get(Submission, "sub_disc")
            assert sub.is_extracted is True
            assert sub.is_release is False

            releases = session.execute(select(Release)).scalars().all()
            assert len(releases) == 0

            artists = session.execute(select(Artist)).scalars().all()
            assert len(artists) == 0

            sub_rels = session.execute(select(SubmissionRelease)).scalars().all()
            assert len(sub_rels) == 0

    def test_atomicity_and_rollback_on_failure(self, session_factory):
        """Failure mid-transaction rolls back all changes, leaving is_extracted=False."""
        with session_factory() as session:
            with session.begin():
                create_test_submission(session, "sub_err", "red_err", "Artist - Album")

        service = ReleasePersistenceService(session_factory)

        # Passing non-existent submission_id should raise ValueError
        with pytest.raises(ValueError):
            service.persist("non_existent_sub", [
                ExtractedRelease(
                    artists=["Ghost Artist"],
                    album_title="Ghost Album",
                )
            ])

        with session_factory() as session:
            # Ghost artist / release must not exist
            artists = session.execute(select(Artist).where(Artist.name == "Ghost Artist")).scalars().all()
            assert len(artists) == 0

    def test_idempotent_reprocessing(self, session_factory):
        """Calling persist multiple times on the same submission succeeds without duplicate rows."""
        with session_factory() as session:
            with session.begin():
                create_test_submission(session, "sub_idem", "red_idem", "Radiohead - Kid A")

        service = ReleasePersistenceService(session_factory)
        ext = ExtractedRelease(
            artists=["Radiohead"],
            combined_artist="Radiohead",
            album_title="Kid A",
            variant="2LP 10-inch",
        )

        res1 = service.persist("sub_idem", [ext])
        res2 = service.persist("sub_idem", [ext])

        assert res1[0].id == res2[0].id

        with session_factory() as session:
            all_releases = session.execute(select(Release)).scalars().all()
            assert len(all_releases) == 1
            all_sub_rels = session.execute(select(SubmissionRelease)).scalars().all()
            assert len(all_sub_rels) == 1

    def test_persist_with_reddit_id_or_fullname_fallback(self, session_factory):
        """Verify persistence resolves submission whether fullname or base36 reddit_id is passed."""
        with session_factory() as session:
            with session.begin():
                create_test_submission(session, "t3_fallback_1", "fallback_1", "Artist 1 - Album 1")
                create_test_submission(session, "fallback_2", "fallback_2", "Artist 2 - Album 2")

        service = ReleasePersistenceService(session_factory)
        ext = ExtractedRelease(artists=["Artist"], album_title="Album")

        # Pass short ID for submission keyed by fullname
        res1 = service.persist("fallback_1", [ext])
        assert len(res1) == 1

        # Pass fullname for submission keyed by short ID
        res2 = service.persist("t3_fallback_2", [ext])
        assert len(res2) == 1

        with session_factory() as session:
            s1 = session.get(Submission, "t3_fallback_1")
            s2 = session.get(Submission, "fallback_2")
            assert s1.is_extracted is True
            assert s2.is_extracted is True
