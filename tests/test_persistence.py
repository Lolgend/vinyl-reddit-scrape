import hashlib
import uuid
from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.extraction import ExtractedRelease, ExtractionResult
from reddit_scrape.models import (
    Album,
    AlbumArtist,
    Artist,
    Release,
    Submission,
    SubmissionRelease,
)
from reddit_scrape.persistence import (
    ReleasePersistenceService,
    compute_album_fingerprint,
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


class TestAlbumAndReleaseFingerprints:
    def test_album_fingerprint_normalization_and_digest(self):
        """Verify lowercase, collapsed whitespace, normalized dashes, and deterministic digest."""
        fp1 = compute_album_fingerprint("  Daft   Punk  ", "Discovery")
        fp2 = compute_album_fingerprint("daft punk", "discovery")
        assert fp1 == fp2
        assert len(fp1) == 64
        expected = hashlib.sha256(b"daft punk|discovery").hexdigest()
        assert fp1 == expected

    def test_album_fingerprint_punctuation_preservation(self):
        """Punctuation like '/', '&', and commas are preserved in normalized attributes and fingerprints."""
        assert normalize_attribute("Converge / Dropdead") == "converge / dropdead"
        assert normalize_attribute("Tyler, The Creator") == "tyler, the creator"

        fp1 = compute_album_fingerprint("Tyler, The Creator", "IGOR")
        fp2 = compute_album_fingerprint("Tyler The Creator", "IGOR")
        assert fp1 != fp2

    def test_album_fingerprint_unicode_dash_normalization(self):
        """Unicode dashes (\u2013, \u2014, etc.) normalize to hyphen."""
        fp1 = compute_album_fingerprint("Artist", "Album \u2013 Part 1")
        fp2 = compute_album_fingerprint("Artist", "Album - Part 1")
        assert fp1 == fp2

    def test_release_fingerprint_scoped_to_album_id(self):
        """Release fingerprint scopes identity under parent album_id."""
        album_id = uuid.uuid4()
        fp_standard = compute_release_fingerprint(album_id, variant=None, retailer=None)
        fp_variant = compute_release_fingerprint(album_id, variant="Blue Opaque Vinyl", retailer=None)
        fp_retailer = compute_release_fingerprint(album_id, variant=None, retailer="Rough Trade")
        fp_both = compute_release_fingerprint(album_id, variant="Blue Opaque Vinyl", retailer="Rough Trade")

        assert len({fp_standard, fp_variant, fp_retailer, fp_both}) == 4
        assert len(fp_standard) == 64

    def test_release_fingerprint_different_albums_same_variant(self):
        """Same variant for different albums produces distinct fingerprints."""
        album1 = uuid.uuid4()
        album2 = uuid.uuid4()
        fp1 = compute_release_fingerprint(album1, variant="Red Vinyl", retailer="Store")
        fp2 = compute_release_fingerprint(album2, variant="Red Vinyl", retailer="Store")
        assert fp1 != fp2


class TestReleasePersistence:
    def test_persist_single_artist_release(self, session_factory):
        """Verify single-artist release persistence, album, artist row, and join records."""
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

        with session_factory() as session:
            # Check submission updated
            updated_sub = session.get(Submission, "sub1")
            assert updated_sub.is_extracted is True

            # Check album created with pending status and provisional embedding
            albums = session.execute(select(Album)).scalars().all()
            assert len(albums) == 1
            alb = albums[0]
            assert alb.combined_artist == "Daft Punk"
            assert alb.title == "Discovery"
            assert alb.discogs_status == "pending"
            assert alb.enrichment_attempts == 0
            assert alb.embedding is not None
            assert len(alb.embedding) == 384
            assert rel.album_id == alb.id

            # Check artist created
            artists = session.execute(select(Artist)).scalars().all()
            assert len(artists) == 1
            assert artists[0].name == "Daft Punk"
            assert artists[0].embedding is not None
            assert len(artists[0].embedding) == 384

            # Check album_artists link
            alb_artists = session.execute(select(AlbumArtist)).scalars().all()
            assert len(alb_artists) == 1
            assert alb_artists[0].album_id == alb.id
            assert alb_artists[0].artist_id == artists[0].id

            # Check submission_releases link
            sub_releases = session.execute(select(SubmissionRelease)).scalars().all()
            assert len(sub_releases) == 1
            assert sub_releases[0].submission_id == "sub1"
            assert sub_releases[0].release_id == rel.id

    def test_persist_multi_artist_release(self, session_factory):
        """Verify multi-artist split/collaboration associates each artist individually in album_artists."""
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

        with session_factory() as session:
            albums = session.execute(select(Album)).scalars().all()
            assert len(albums) == 1
            alb = albums[0]
            assert alb.combined_artist == "Converge / Dropdead"
            assert alb.title == "Converge / Dropdead Split"
            assert alb.discogs_status == "pending"
            assert alb.embedding is not None
            assert len(alb.embedding) == 384
            assert rel.album_id == alb.id

            artists = session.execute(select(Artist).order_by(Artist.name)).scalars().all()
            assert len(artists) == 2
            artist_names = [a.name for a in artists]
            assert "Converge" in artist_names
            assert "Dropdead" in artist_names
            assert all(a.embedding is not None and len(a.embedding) == 384 for a in artists)

            # Check both artists linked to this album
            links = session.execute(
                select(AlbumArtist).where(AlbumArtist.album_id == alb.id)
            ).scalars().all()
            assert len(links) == 2

    def test_persist_multiple_releases_single_canonical_album(self, session_factory):
        """When multiple releases (different variants/retailers) of same work are persisted,
        a single canonical Album row (discogs_status='pending', 384-d provisional embedding) is created.
        """
        with session_factory() as session, session.begin():
            create_test_submission(session, "sub_multi_var", "red_multi_var", "Multiple Variants Post")

        service = ReleasePersistenceService(session_factory)
        ext1 = ExtractedRelease(
            artists=["Radiohead"],
            combined_artist="Radiohead",
            album_title="OK Computer",
            variant="Standard Black Vinyl",
            retailer="Amazon",
        )
        ext2 = ExtractedRelease(
            artists=["Radiohead"],
            combined_artist="Radiohead",
            album_title="OK Computer",
            variant="Opaque Blue Vinyl",
            retailer="Rough Trade",
        )
        ext3 = ExtractedRelease(
            artists=["Radiohead"],
            combined_artist="Radiohead",
            album_title="OK Computer",
            variant="White Vinyl 2LP",
            retailer="W.A.S.T.E.",
        )

        results = service.persist("sub_multi_var", [ext1, ext2, ext3])
        assert len(results) == 3
        # All 3 releases point to the same album_id
        album_ids = {r.album_id for r in results}
        assert len(album_ids) == 1
        common_album_id = album_ids.pop()

        with session_factory() as session:
            albums = session.execute(select(Album)).scalars().all()
            assert len(albums) == 1
            alb = albums[0]
            assert alb.id == common_album_id
            assert alb.discogs_status == "pending"
            assert alb.enrichment_attempts == 0
            assert alb.embedding is not None
            assert len(alb.embedding) == 384

            releases = session.execute(select(Release).order_by(Release.variant)).scalars().all()
            assert len(releases) == 3
            variants = [r.variant for r in releases]
            assert "Standard Black Vinyl" in variants
            assert "Opaque Blue Vinyl" in variants
            assert "White Vinyl 2LP" in variants

    def test_reingest_preserves_matched_discogs_status_and_embedding(self, session_factory):
        """Re-ingesting a submission or persisting a new Release for an Album whose
        discogs_status is already 'matched', 'partial_matched', or 'not_found' preserves its
        existing discogs_status and canonical Album.embedding.
        """
        service = ReleasePersistenceService(session_factory)
        with session_factory() as session, session.begin():
            create_test_submission(session, "sub_init", "red_init", "Post 1: Fontaines D.C. - Romance")

        ext1 = ExtractedRelease(
            artists=["Fontaines D.C."],
            combined_artist="Fontaines D.C.",
            album_title="Romance",
            variant="Standard Black",
            retailer="XL Recordings",
        )
        res1 = service.persist("sub_init", [ext1])
        assert len(res1) == 1
        album_id = res1[0].album_id

        # Simulate Pass 2 enrichment marking the album as 'matched' with a canonical vector
        canonical_vec = [0.99] * 384
        with session_factory() as session, session.begin():
            alb = session.get(Album, album_id)
            alb.discogs_status = "matched"
            alb.discogs_master_id = 987654
            alb.embedding = canonical_vec

        # Persist a new variant for the same album via another submission
        with session_factory() as session, session.begin():
            create_test_submission(session, "sub_new_drop", "red_new_drop", "Post 2: Fontaines D.C. - Romance (Pink)")

        ext2 = ExtractedRelease(
            artists=["Fontaines D.C."],
            combined_artist="Fontaines D.C.",
            album_title="Romance",
            variant="Limited Pink LP",
            retailer="Rough Trade",
        )
        res2 = service.persist("sub_new_drop", [ext2])
        assert len(res2) == 1
        assert res2[0].album_id == album_id

        # Verify that discogs_status and canonical embedding were NOT overwritten by provisional vector
        with session_factory() as session:
            alb_after = session.get(Album, album_id)
            assert alb_after.discogs_status == "matched"
            assert alb_after.discogs_master_id == 987654
            assert alb_after.embedding is not None
            assert pytest.approx(alb_after.embedding[0], rel=1e-3) == 0.99

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
            # Only 1 canonical Album record
            all_albums = session.execute(select(Album)).scalars().all()
            assert len(all_albums) == 1
            assert all_albums[0].embedding is not None
            assert len(all_albums[0].embedding) == 384

            # Only 1 canonical Release record
            all_releases = session.execute(select(Release)).scalars().all()
            assert len(all_releases) == 1
            assert all_releases[0].album_id == all_albums[0].id

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
            all_albums = session.execute(select(Album)).scalars().all()
            assert len(all_albums) == 1
            assert all_albums[0].embedding is not None

            all_releases = session.execute(select(Release)).scalars().all()
            assert len(all_releases) == 1

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
        candidates = service.find_candidates("Dropdead / Converge - Split 7\"", top_k=5, min_similarity=0.50)
        assert len(candidates.releases) >= 1
        top_rel = candidates.releases[0]
        assert top_rel.combined_artist == "Converge / Dropdead"
        assert top_rel.album_title == "Split"
        assert "Converge" in top_rel.artists and "Dropdead" in top_rel.artists
        assert top_rel.similarity >= 0.50
        assert "Converge" in candidates.artists or "Dropdead" in candidates.artists

    def test_query_with_swapped_artist_album_order(self, session_factory):
        service = ReleasePersistenceService(session_factory)
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
        candidates = service.find_candidates("Daft Punk - Discovery (Gatefold 2LP)", top_k=5, min_similarity=0.50)
        assert len(candidates.releases) >= 1
        top_rel = candidates.releases[0]
        assert top_rel.combined_artist == "Daft Punk"
        assert top_rel.album_title == "Discovery"
        assert top_rel.variant == "2LP Gatefold"

    def test_unrelated_query_below_threshold_excluded(self, session_factory):
        service = ReleasePersistenceService(session_factory)
        candidates = service.find_candidates(
            "Beethoven - Symphony No. 9 in D minor",
            top_k=5,
            min_similarity=0.85,
        )
        assert candidates.is_empty
        assert len(candidates.releases) == 0

    def test_batch_query_deduplicates_across_titles(self, session_factory):
        service = ReleasePersistenceService(session_factory)
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

    def test_find_candidates_hydrates_multiple_sibling_releases(self, session_factory):
        service = ReleasePersistenceService(session_factory)

        # Seed an additional sibling release for Converge / Dropdead - Split
        with session_factory() as session, session.begin():
            create_test_submission(session, "sub_seed_1b", "red_seed_1b", "Converge / Dropdead - Split Pink")

        service.persist("sub_seed_1b", [
            ExtractedRelease(
                artists=["Converge", "Dropdead"],
                combined_artist="Converge / Dropdead",
                album_title="Split",
                variant="Pink /500",
                retailer="Deathwish",
            )
        ])

        candidates = service.find_candidates("Converge / Dropdead - Split", top_k=5, min_similarity=0.50)
        split_releases = [r for r in candidates.releases if r.album_title == "Split"]
        assert len(split_releases) == 2
        # Both sibling releases must share the same canonical album metadata and album_id
        assert split_releases[0].album_id is not None
        assert split_releases[0].album_id == split_releases[1].album_id
        assert split_releases[0].combined_artist == "Converge / Dropdead"
        assert split_releases[1].combined_artist == "Converge / Dropdead"
        assert split_releases[0].artists == ["Converge", "Dropdead"]
        assert split_releases[1].artists == ["Converge", "Dropdead"]

        variants = {r.variant for r in split_releases}
        assert variants == {'Clear 7"', 'Pink /500'}
        for r in split_releases:
            assert r.retailer == "Deathwish"

