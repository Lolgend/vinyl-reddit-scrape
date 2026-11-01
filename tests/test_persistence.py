from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.extraction import ExtractedRelease, ExtractionResult
from reddit_scrape.models import (
    Artist,
    Release,
    ReleaseArtist,
    Submission,
    SubmissionRelease,
)
from reddit_scrape.persistence import (
    ReleasePersistenceService,
    compute_release_fingerprint,
    normalize_attribute,
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
        with session_factory() as session, session.begin():
            create_test_submission(session, "sub1", "red1", "Daft Punk - Discovery")

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
        assert rel.embedding is not None
        assert len(rel.embedding) == 384

        with session_factory() as session:
            # Check submission updated
            updated_sub = session.get(Submission, "sub1")
            assert updated_sub.is_extracted is True

            # Check artist created
            artists = session.execute(select(Artist)).scalars().all()
            assert len(artists) == 1
            assert artists[0].name == "Daft Punk"
            assert artists[0].embedding is not None
            assert len(artists[0].embedding) == 384

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
        with session_factory() as session, session.begin():
            create_test_submission(session, "sub2", "red2", "Converge / Dropdead Split")

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
        assert rel.embedding is not None
        assert len(rel.embedding) == 384

        with session_factory() as session:
            artists = session.execute(select(Artist).order_by(Artist.name)).scalars().all()
            assert len(artists) == 2
            artist_names = [a.name for a in artists]
            assert "Converge" in artist_names
            assert "Dropdead" in artist_names
            assert all(a.embedding is not None and len(a.embedding) == 384 for a in artists)

            # Check both artists linked to this release
            links = session.execute(
                select(ReleaseArtist).where(ReleaseArtist.release_id == rel.id)
            ).scalars().all()
            assert len(links) == 2

    def test_deduplication_two_submissions_same_release(self, session_factory):
        """Two different submissions for the exact same pressing resolve to the same Release row."""
        with session_factory() as session, session.begin():
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
            assert all_releases[0].embedding is not None
            assert len(all_releases[0].embedding) == 384

            # Only 1 canonical Artist record
            all_artists = session.execute(select(Artist)).scalars().all()
            assert len(all_artists) == 1
            assert all_artists[0].embedding is not None
            assert len(all_artists[0].embedding) == 384

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
        with session_factory() as session, session.begin():
            create_test_submission(session, "sub_disc", "red_disc", "[Discussion] Favorite turntables?")

        service = ReleasePersistenceService(session_factory)
        ext_result = ExtractionResult(is_release=False, confidence="high", releases=[])
        result = service.persist("sub_disc", ext_result)

        assert result == []

        with session_factory() as session:
            sub = session.get(Submission, "sub_disc")
            assert sub.is_extracted is True
            assert sub.is_release is False

            all_releases = session.execute(select(Release)).scalars().all()
            assert len(all_releases) == 0

    def test_unknown_submission_raises_value_error(self, session_factory):
        service = ReleasePersistenceService(session_factory)
        ext_release = ExtractedRelease(
            artists=["Daft Punk"],
            combined_artist="Daft Punk",
            album_title="Discovery",
        )
        with pytest.raises(ValueError, match="does not exist"):
            service.persist("non_existent_id", [ext_release])

    def test_idempotent_reingestion(self, session_factory):
        """Re-ingesting the exact same submission multiple times is safe and maintains consistency."""
        with session_factory() as session, session.begin():
            create_test_submission(session, "sub_dup", "red_dup", "Daft Punk - Homework")

        service = ReleasePersistenceService(session_factory)
        ext = ExtractedRelease(
            artists=["Daft Punk"],
            combined_artist="Daft Punk",
            album_title="Homework",
            variant="Standard Black",
            retailer="Amazon",
        )

        res1 = service.persist("sub_dup", [ext])
        res2 = service.persist("sub_dup", [ext])

        assert len(res1) == 1
        assert len(res2) == 1
        assert res1[0].id == res2[0].id

        with session_factory() as session:
            all_releases = session.execute(select(Release)).scalars().all()
            assert len(all_releases) == 1
            assert all_releases[0].embedding is not None
            assert len(all_releases[0].embedding) == 384

            all_sub_rels = session.execute(select(SubmissionRelease)).scalars().all()
            assert len(all_sub_rels) == 1

    def test_atomicity_and_rollback_on_failure(self, session_factory):
        """Failure mid-transaction rolls back all changes, leaving is_extracted=False."""
        with session_factory() as session, session.begin():
            create_test_submission(session, "sub_err", "red_err", "Artist - Album")

        service = ReleasePersistenceService(session_factory)

        # Passing non-existent submission_id should raise ValueError
        with pytest.raises(ValueError, match="does not exist"):
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

    def test_persist_with_reddit_id_or_fullname_fallback(self, session_factory):
        """Verify persistence resolves submission whether fullname or base36 reddit_id is passed."""
        with session_factory() as session, session.begin():
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
            assert s1 is not None and s1.is_extracted is True
            assert s2 is not None and s2.is_extracted is True


class TestCatalogCandidateRetrievalIntegration:
    @pytest.fixture(autouse=True)
    def seed_catalog(self, session_factory):
        service = ReleasePersistenceService(session_factory)
        with session_factory() as session, session.begin():
            create_test_submission(session, "sub_seed_1", "red_seed_1", "Converge / Dropdead - Split")
            create_test_submission(session, "sub_seed_2", "red_seed_2", "Daft Punk - Discovery")
            create_test_submission(session, "sub_seed_3", "red_seed_3", "Radiohead - OK Computer")

        # Persist canonical releases
        service.persist("sub_seed_1", [
            ExtractedRelease(
                artists=["Converge", "Dropdead"],
                combined_artist="Converge / Dropdead",
                album_title="Split",
                variant="Clear 7\"",
                retailer="Deathwish",
            )
        ])
        service.persist("sub_seed_2", [
            ExtractedRelease(
                artists=["Daft Punk"],
                combined_artist="Daft Punk",
                album_title="Discovery",
                variant="2LP Gatefold",
                retailer="Virgin",
            )
        ])
        service.persist("sub_seed_3", [
            ExtractedRelease(
                artists=["Radiohead"],
                combined_artist="Radiohead",
                album_title="OK Computer",
                variant="180g Vinyl",
                retailer=None,
            )
        ])

    def test_query_with_reversed_billing_order(self, session_factory):
        service = ReleasePersistenceService(session_factory)
        # Reversed billing order: "Dropdead / Converge - Split 7\""
        candidates = service.find_candidates("Dropdead / Converge - Split 7\"", top_k=5, min_similarity=0.50)
        assert len(candidates.releases) >= 1
        top_rel = candidates.releases[0]
        assert top_rel.combined_artist == "Converge / Dropdead"
        assert top_rel.album_title == "Split"
        assert "Converge" in top_rel.artists and "Dropdead" in top_rel.artists
        assert top_rel.similarity >= 0.50
        # Contributing artists also retrieved in artists
        assert "Converge" in candidates.artists or "Dropdead" in candidates.artists

    def test_query_with_swapped_artist_album_order(self, session_factory):
        service = ReleasePersistenceService(session_factory)
        # Swapped artist/album: "Discovery - Daft Punk"
        candidates = service.find_candidates("Discovery - Daft Punk", top_k=5, min_similarity=0.50)
        assert len(candidates.releases) >= 1
        top_rel = candidates.releases[0]
        assert top_rel.combined_artist == "Daft Punk"
        assert top_rel.album_title == "Discovery"
        assert top_rel.artists == ["Daft Punk"]
        assert top_rel.similarity >= 0.50
        assert "Daft Punk" in candidates.artists

    def test_query_with_reordered_variant_tokens(self, session_factory):
        service = ReleasePersistenceService(session_factory)
        # Reordered variant tokens
        candidates = service.find_candidates("Daft Punk - Discovery (Gatefold 2LP)", top_k=5, min_similarity=0.50)
        assert len(candidates.releases) >= 1
        top_rel = candidates.releases[0]
        assert top_rel.combined_artist == "Daft Punk"
        assert top_rel.album_title == "Discovery"
        assert top_rel.variant == "2LP Gatefold"

    def test_unrelated_query_below_threshold_excluded(self, session_factory):
        service = ReleasePersistenceService(session_factory)
        # High threshold for unrelated query
        candidates = service.find_candidates(
            "Beethoven - Symphony No. 9 in D minor",
            top_k=5,
            min_similarity=0.85,
        )
        assert candidates.is_empty
        assert len(candidates.releases) == 0

    def test_batch_query_deduplicates_across_titles(self, session_factory):
        service = ReleasePersistenceService(session_factory)
        # Two queries matching the same canonical release
        candidates = service.find_candidates([
            "Daft Punk - Discovery (Gatefold)",
            "Discovery by Daft Punk",
        ], top_k=5, min_similarity=0.50)

        # Release ID must be unique in candidates.releases
        rel_ids = [r.id for r in candidates.releases]
        assert len(rel_ids) == len(set(rel_ids))
        # Artist names must be unique
        assert len(candidates.artists) == len(set(candidates.artists))
        assert any(r.combined_artist == "Daft Punk" for r in candidates.releases)
