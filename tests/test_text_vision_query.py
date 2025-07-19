import numpy as np
import pandas as pd
import pytest

import lazyslide as zs


class TestTextEmbedding:
    """Tests for the text_embedding function."""

    def test_text_embedding_plip(self):
        """Test text_embedding with PLIP model."""
        texts = ["This is a test", "Another test text"]
        embeddings = zs.tl.text_embedding(texts, model="plip")

        # Check that the output is a DataFrame with the expected shape and index
        assert isinstance(embeddings, pd.DataFrame)
        assert embeddings.shape[0] == 2  # Two texts
        assert list(embeddings.index) == texts

        # Check that the embeddings are non-zero
        assert not np.allclose(embeddings.values, 0)

    def test_text_embedding_invalid_model(self):
        """Test text_embedding with an invalid model."""
        texts = ["This is a test", "Another test text"]
        with pytest.raises(ValueError, match="Invalid model"):
            zs.tl.text_embedding(texts, model="invalid_model")


class TestTextImageSimilarity:
    """Tests for the text_image_similarity function."""

    @pytest.fixture(autouse=True)
    def setup(self, wsi_small):
        """Setup for text_image_similarity tests."""
        zs.tl.feature_extraction(wsi_small, model="plip")

        # Verify that the tiles were created
        assert "plip_tiles" in wsi_small.tables
        assert len(wsi_small.tables["plip_tiles"]) > 0

        # Define test texts
        self.texts = ["This is a test", "Another test text"]

        # Store the WSI and tile_key for use in tests
        self.wsi = wsi_small

    def test_text_image_similarity_plip(self):
        """Test text_image_similarity with PLIP model."""
        # Get text embeddings
        text_embeddings = zs.tl.text_embedding(self.texts, model="plip")

        # Compute similarity
        zs.tl.text_image_similarity(
            self.wsi, text_embeddings, model="plip", key_added="plip_similarity"
        )

        # Check that the similarity scores were added to the WSI
        assert "plip_similarity" in self.wsi.tables

        # Check that the similarity scores have the expected shape
        similarity_table = self.wsi.tables["plip_similarity"]
        assert similarity_table.X.shape[0] == len(self.wsi.tables["plip_tiles"])
        assert similarity_table.X.shape[1] == len(self.texts)

        # Check that the variable names match the text embeddings index
        assert list(similarity_table.var.index) == self.texts

    def test_text_image_similarity_with_softmax(self):
        """Test text_image_similarity with softmax applied."""
        # Get text embeddings
        text_embeddings = zs.tl.text_embedding(self.texts, model="plip")

        # Compute similarity with softmax
        zs.tl.text_image_similarity(
            self.wsi,
            text_embeddings,
            model="plip",
            key_added="plip_similarity_softmax",
            softmax=True,
        )

        # Check that the similarity scores were added to the WSI
        assert "plip_similarity_softmax" in self.wsi.tables

        # Check that the similarity scores have the expected shape
        similarity_table = self.wsi.tables["plip_similarity_softmax"]
        assert similarity_table.X.shape[0] == len(self.wsi.tables["plip_tiles"])
        assert similarity_table.X.shape[1] == len(self.texts)

        # Check that the scores are probabilities (sum to 1 for each tile)
        row_sums = similarity_table.X.sum(axis=1)
        assert np.allclose(row_sums, 1.0, rtol=1e-5)

    def test_text_image_similarity_with_custom_scoring(self):
        """Test text_image_similarity with a custom scoring function."""
        # Get text embeddings
        text_embeddings = zs.tl.text_embedding(self.texts, model="plip")

        # Define a custom scoring function (cosine similarity)
        def cosine_similarity(X, Y):
            # Normalize X and Y
            X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
            Y_norm = Y / np.linalg.norm(Y, axis=0, keepdims=True)
            return np.dot(X_norm, Y_norm)

        # Compute similarity with custom scoring function
        zs.tl.text_image_similarity(
            self.wsi,
            text_embeddings,
            model="plip",
            key_added="plip_similarity_cosine",
            scoring_func=cosine_similarity,
        )

        # Check that the similarity scores were added to the WSI
        assert "plip_similarity_cosine" in self.wsi.tables

        # Check that the similarity scores have the expected shape
        similarity_table = self.wsi.tables["plip_similarity_cosine"]
        assert similarity_table.X.shape[0] == len(self.wsi.tables["plip_tiles"])
        assert similarity_table.X.shape[1] == len(self.texts)

        # Check that the scores are between -1 and 1 (cosine similarity range)
        assert np.all(similarity_table.X >= -1.0)
        assert np.all(similarity_table.X <= 1.0)

    def test_text_image_similarity_invalid_scoring_func(self):
        """Test text_image_similarity with an invalid scoring function."""
        # Get text embeddings
        text_embeddings = zs.tl.text_embedding(self.texts, model="plip")

        # Define an invalid scoring function that will raise an error
        def invalid_scoring_func(X, Y):
            # This will raise an error because the shapes don't match
            return X + Y

        # Compute similarity with invalid scoring function
        with pytest.raises(ValueError, match="Error in custom scoring_func"):
            zs.tl.text_image_similarity(
                self.wsi,
                text_embeddings,
                model="plip",
                scoring_func=invalid_scoring_func,
            )
