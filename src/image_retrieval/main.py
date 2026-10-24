"""The CLI is the sole corpus writer."""

import argparse
import logging
import sys
from contextlib import closing

from tqdm import tqdm

from .config import load_settings
from .features import FeatureStore, writer_lock
from .indexing import IndexReport, encode_and_index
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
            failure_log = path.with_name("indexing-failures.log")
            print(f"Image failure log: {failure_log}", file=sys.stderr)
            with (
                closing(logging.FileHandler(failure_log, encoding="utf-8")) as handler,
                tqdm(desc="Indexing images", unit="image", dynamic_ncols=True) as bar,
            ):
                # A local logger avoids changing application/global logging handlers.
                logger = logging.Logger("image_retrieval.indexing.failures")
                handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
                )
                logger.addHandler(handler)

                def skipped(message: str) -> None:
                    # Called inside the decoder's exception handler: retain traceback.
                    logger.error(
                        "corpus=%s root=%s %s",
                        corpus.id,
                        corpus.root,
                        message,
                        exc_info=True,
                    )
                    handler.flush()
                    tqdm.write(message, file=sys.stderr)

                def progress(completed: int, total: int, current: IndexReport) -> None:
                    bar.total = total
                    bar.set_postfix(
                        encoded=current.generated,
                        reused=current.reused,
                        skipped=len(current.skipped),
                        warnings=current.metadata_warnings,
                        refresh=False,
                    )
                    bar.update(completed - bar.n)

                def metadata_warning(message: str) -> None:
                    logger.warning(
                        "corpus=%s root=%s %s", corpus.id, corpus.root, message
                    )
                    handler.flush()
                    tqdm.write(message, file=sys.stderr)

                report = encode_and_index(
                    embedder,
                    corpus.root,
                    store,
                    batch_size=settings.runtime.batch_size,
                    checkpoint_interval=settings.runtime.checkpoint_interval,
                    report_skip=skipped,
                    report_progress=progress,
                    report_warning=metadata_warning,
                )
        print(
            f"Complete: {len(store.keys)} paths; {report.generated} generated, {report.reused} reused, {len(report.skipped)} skipped, {report.metadata_warnings} metadata warnings.\nSnapshot: {path}"
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
