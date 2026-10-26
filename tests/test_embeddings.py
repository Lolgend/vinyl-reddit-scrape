from fastembed import TextEmbedding

from reddit_scrape.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    compute_artist_text,
    compute_release_text,
    embed_texts,
    get_embedding_model,
)


class TestEmbeddingTextFormatting:
    def test_compute_release_text_basic(self):
        text = compute_release_text(combined_artist="Daft Punk", album_title="Discovery")
        assert text == "Daft Punk - Discovery"

    def test_compute_release_text_with_variant(self):
        text = compute_release_text(
            combined_artist="Daft Punk",
            album_title="Discovery",
            variant="Limited Edition Red /500",
        )
        assert text == "Daft Punk - Discovery (Limited Edition Red /500)"

    def test_compute_release_text_with_retailer(self):
        text = compute_release_text(
            combined_artist="Daft Punk",
            album_title="Discovery",
            retailer="Rough Trade",
        )
        assert text == "Daft Punk - Discovery [Rough Trade]"

    def test_compute_release_text_with_variant_and_retailer(self):
        text = compute_release_text(
            combined_artist="Daft Punk",
            album_title="Discovery",
            variant="2LP Gatefold",
            retailer="Rough Trade",
        )
        assert text == "Daft Punk - Discovery (2LP Gatefold) [Rough Trade]"

    def test_compute_release_text_whitespace_handling(self):
        text = compute_release_text(
            combined_artist="  Daft Punk  ",
            album_title="  Discovery  ",
            variant="   ",
            retailer="  Target  ",
        )
        assert text == "Daft Punk - Discovery [Target]"

    def test_compute_artist_text(self):
        assert compute_artist_text("  Tyler, The Creator  ") == "Tyler, The Creator"


class TestEmbeddingModel:
    def test_get_embedding_model_lazy_singleton(self):
        model1 = get_embedding_model()
        model2 = get_embedding_model()
        assert isinstance(model1, TextEmbedding)
        assert model1 is model2
        assert model1.model_name == DEFAULT_EMBEDDING_MODEL
        assert DEFAULT_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
        assert EMBEDDING_DIM == 384

    def test_embed_texts_returns_correct_dimensions(self):
        texts = ["Daft Punk - Discovery", "Radiohead - OK Computer"]
        embeddings = embed_texts(texts)
        assert len(embeddings) == 2
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == EMBEDDING_DIM
            assert all(isinstance(val, float) for val in emb)

    def test_embed_texts_empty(self):
        assert embed_texts([]) == []
