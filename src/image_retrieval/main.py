"""The CLI is the sole corpus writer."""

import argparse
import sys

from .config import load_settings
from .features import FeatureStore, writer_lock
from .indexing import encode_and_index
from .model import create_embedder


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate local corpus embeddings from profiles.yaml"
    )
    parser.add_argument(
        "--config", help="YAML path (repository-relative unless absolute)"
    )
    parser.add_argument("--profile", help="Named model profile")
    parser.add_argument("--corpus", help="Named image collection")
    args = parser.parse_args(argv)
    embedder = None
    try:
        settings = load_settings(args.config)
        profile, corpus = settings.select(args.profile, args.corpus)
        embedder = create_embedder(profile, settings.runtime)
        path = settings.store_path(corpus, embedder.recipe_id)
        with writer_lock(path):
            store = FeatureStore.load(
                path, corpus.id, embedder.manifest, missing_ok=True
            )
            report = encode_and_index(
                embedder,
                corpus.root,
                store,
                batch_size=settings.runtime.batch_size,
                checkpoint_interval=settings.runtime.checkpoint_interval,
                report_skip=lambda message: print(message, file=sys.stderr),
            )
        print(
            f"Complete: {len(store.keys)} paths; {report.generated} generated, {report.reused} reused, {len(report.skipped)} skipped.\nSnapshot: {path}"
        )
        return 2 if report.skipped else 0
    except KeyboardInterrupt:
        print("Interrupted; validated progress saved as incomplete.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Indexing failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if embedder is not None:
            embedder.close()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
