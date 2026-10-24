"""Strict explicit settings; configured paths are repository-relative."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from yaml.resolver import BaseResolver


class ConfigError(ValueError):
    pass


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ConfigError("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _fields(value: Any, required: set[str]):
    if not isinstance(value, dict) or value.keys() != required:
        raise ConfigError(f"Expected exactly these keys: {sorted(required)}")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a nonempty string")
    return value


def _positive(value: Any, label: str, maximum: int = 1_000_000) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ConfigError(f"{label} must be an integer in 1..{maximum}")
    return value


@dataclass(frozen=True)
class Profile:
    family: str
    model: str
    revision: str
    processor_revision: str


@dataclass(frozen=True)
class Corpus:
    id: str
    root: Path


@dataclass(frozen=True)
class Runtime:
    device: str = "auto"
    precision: str = "float32"
    batch_size: int = 1
    offline: bool = False
    checkpoint_interval: int = 500


@dataclass(frozen=True)
class Settings:
    config_path: Path
    repo_root: Path
    profiles: dict[str, Profile]
    corpora: dict[str, Corpus]
    default_profile: str
    default_corpus: str
    runtime: Runtime
    cache_root: Path

    def select(self, profile: str | None = None, corpus: str | None = None):
        profile = profile or self.default_profile
        corpus = corpus or self.default_corpus
        if profile not in self.profiles or corpus not in self.corpora:
            raise ConfigError("Unknown profile or corpus selection")
        return self.profiles[profile], self.corpora[corpus]

    def store_path(self, corpus: Corpus, recipe_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", recipe_id):
            raise ConfigError("Invalid recipe fingerprint")
        return self.cache_root / corpus.id / recipe_id / "features.npz"


def repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ConfigError("Cannot locate repository root; use a repository installation")


def load_settings(
    path: str | Path | None = None, *, repo_root: Path | None = None
) -> Settings:
    root = (repo_root or repository_root()).resolve()
    config_path = Path(path) if path is not None else Path("profiles.yaml")
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        data = yaml.load(config_path.read_text(), Loader=_UniqueLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read configuration {config_path}: {exc}") from exc
    _fields(
        data,
        {
            "schema_version",
            "default_profile",
            "default_corpus",
            "profiles",
            "corpora",
            "runtime",
            "cache_root",
        },
    )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ConfigError("Unsupported configuration schema_version")
    environment = {**dotenv_values(root / ".env", interpolate=False), **os.environ}

    def resolve(value: Any) -> Path:
        text = _string(value, "path")

        def substitute(match):
            replacement = environment.get(match[1])
            if not replacement:
                raise ConfigError(f"Missing path placeholder ${{{match[1]}}}")
            return replacement

        text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", substitute, text)
        if "$" in text:
            raise ConfigError("Unsupported or unresolved path placeholder")
        result = Path(text).expanduser()
        return (result if result.is_absolute() else root / result).resolve()

    profiles = {}
    if not isinstance(data["profiles"], dict) or not data["profiles"]:
        raise ConfigError("profiles must be a nonempty mapping")
    supported = {
        "embeddinggemma2": "google/embeddinggemma-2",
        "siglip2": "google/siglip2-so400m-patch14-384",
    }
    for name, entry in data["profiles"].items():
        _string(name, "profile name")
        _fields(entry, {"family", "model", "revision", "processor_revision"})
        family = _string(entry["family"], "family")
        if family not in supported or entry["model"] != supported[family]:
            raise ConfigError(
                "Unsupported family/checkpoint; adding models requires an adapter review"
            )
        for field in ("revision", "processor_revision"):
            if not isinstance(entry[field], str) or not re.fullmatch(
                r"[0-9a-f]{40}", entry[field]
            ):
                raise ConfigError(
                    f"{field} must be an immutable 40-character Hub commit"
                )
        if (
            family == "embeddinggemma2"
            and entry["revision"] != entry["processor_revision"]
        ):
            raise ConfigError(
                "EmbeddingGemma model and processor must share a pinned revision"
            )
        profiles[name] = Profile(**entry)
    corpora = {}
    ids = set()
    if not isinstance(data["corpora"], dict) or not data["corpora"]:
        raise ConfigError("corpora must be a nonempty mapping")
    for name, entry in data["corpora"].items():
        _string(name, "corpus name")
        _fields(entry, {"id", "root"})
        identifier = _string(entry["id"], "corpus id")
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier)
            or identifier in ids
        ):
            raise ConfigError("Corpus IDs must be unique, path-safe identifiers")
        ids.add(identifier)
        corpora[name] = Corpus(identifier, resolve(entry["root"]))
    runtime_data = _fields(
        data["runtime"],
        {"device", "precision", "batch_size", "offline", "checkpoint_interval"},
    )
    if runtime_data["device"] not in ("auto", "mps", "cpu"):
        raise ConfigError("device must be auto, mps, or cpu")
    if runtime_data["precision"] != "float32":
        raise ConfigError("Only float32 inference is supported")
    _positive(runtime_data["batch_size"], "batch_size", 256)
    _positive(runtime_data["checkpoint_interval"], "checkpoint_interval")
    if type(runtime_data["offline"]) is not bool:
        raise ConfigError("offline must be a boolean")
    settings = Settings(
        config_path.resolve(),
        root,
        profiles,
        corpora,
        _string(data["default_profile"], "default_profile"),
        _string(data["default_corpus"], "default_corpus"),
        Runtime(**runtime_data),
        resolve(data["cache_root"]),
    )
    settings.select()
    return settings
