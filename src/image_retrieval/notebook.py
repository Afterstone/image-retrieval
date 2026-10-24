"""Read-only notebook lifecycle. No indexing or corpus publication methods."""

from .config import Settings
from .features import FeatureStore
from .model import Embedder, create_embedder


class NotebookSession:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder: Embedder | None = None
        self.store: FeatureStore | None = None
        self.selection: tuple[str, str] | None = None

    def select(self, profile: str, corpus: str) -> None:
        if self.selection == (profile, corpus):
            return
        self.store = None
        # Release before creating the replacement; session is the sole model owner.
        if self.embedder is not None:
            self.embedder.close()
            self.embedder = None
        self.selection = None
        selected_profile, _ = self.settings.select(profile, corpus)
        self.embedder = create_embedder(selected_profile, self.settings.runtime)
        self.selection = (profile, corpus)

    def reload(self) -> FeatureStore:
        if self.selection is None or self.embedder is None:
            raise ValueError("Select a profile and collection first")
        self.store = None
        _, corpus = self.settings.select(*self.selection)
        self.store = FeatureStore.load(
            self.settings.store_path(corpus, self.embedder.recipe_id),
            corpus.id,
            self.embedder.manifest,
        )
        return self.store

    def indexing_guidance(self) -> str:
        import shlex

        if self.selection is None:
            return "Select a model profile and collection."
        profile, corpus = self.selection
        return " ".join(
            map(
                shlex.quote,
                [
                    "image-retrieval",
                    "--config",
                    str(self.settings.config_path),
                    "--profile",
                    profile,
                    "--corpus",
                    corpus,
                ],
            )
        )
