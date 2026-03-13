#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path("/srv/shared/ICRA2026/datasets/airoa-moma")
DEFAULT_DEST = Path("/srv/shared/ICRA2026/datasets/airoa-moma-pi05")
ACTION_SOURCE_KEY = "action.relative"
ACTION_TARGET_KEY = "action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a pi05-compatible copy of the airoa-moma dataset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination directory if it already exists.",
    )
    return parser.parse_args()


def validate_source(source: Path) -> None:
    required = [
        source / "meta" / "info.json",
        source / "meta" / "stats.json",
        source / "data",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset paths: {missing}")


def copy_tree_with_symlinks(source: Path, dest: Path) -> None:
    def ignore(dirpath: str, names: list[str]) -> set[str]:
        ignored = set()
        rel = Path(dirpath).resolve().relative_to(source.resolve())
        if rel == Path("."):
            if "data" in names:
                ignored.add("data")
            if "meta" in names:
                ignored.add("meta")
        return ignored

    shutil.copytree(source, dest, symlinks=True, ignore=ignore)


def update_info_json(source: Path, dest: Path) -> None:
    with (source / "meta" / "info.json").open() as f:
        info = json.load(f)

    features = info["features"]
    action_feature = features.pop(ACTION_SOURCE_KEY)
    features[ACTION_TARGET_KEY] = action_feature

    dest_meta = dest / "meta"
    dest_meta.mkdir(parents=True, exist_ok=True)
    with (dest_meta / "info.json").open("w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def update_stats_json(source: Path, dest: Path) -> None:
    with (source / "meta" / "stats.json").open() as f:
        stats = json.load(f)

    action_stats = stats.pop(ACTION_SOURCE_KEY)
    stats[ACTION_TARGET_KEY] = action_stats

    dest_meta = dest / "meta"
    dest_meta.mkdir(parents=True, exist_ok=True)
    with (dest_meta / "stats.json").open("w") as f:
        json.dump(stats, f, indent=4)
        f.write("\n")


def copy_meta_extras(source: Path, dest: Path) -> None:
    source_meta = source / "meta"
    dest_meta = dest / "meta"
    dest_meta.mkdir(parents=True, exist_ok=True)

    for path in source_meta.iterdir():
        if path.name in {"info.json", "stats.json"}:
            continue
        target = dest_meta / path.name
        if path.is_dir():
            shutil.copytree(path, target, symlinks=True)
        else:
            shutil.copy2(path, target)


def transform_table(table: pa.Table) -> pa.Table:
    if ACTION_TARGET_KEY in table.column_names:
        return table
    if ACTION_SOURCE_KEY not in table.column_names:
        raise KeyError(f"{ACTION_SOURCE_KEY} not found in parquet columns: {table.column_names}")

    columns = []
    names = []
    inserted = False
    source_column = table[ACTION_SOURCE_KEY]
    source_field = table.schema.field(ACTION_SOURCE_KEY)

    for field in table.schema:
        if field.name == ACTION_SOURCE_KEY:
            columns.append(source_column)
            names.append(ACTION_TARGET_KEY)
            inserted = True
        else:
            columns.append(table[field.name])
            names.append(field.name)

    if not inserted:
        raise RuntimeError(f"Failed to map {ACTION_SOURCE_KEY} to {ACTION_TARGET_KEY}")

    metadata = dict(table.schema.metadata or {})
    metadata[b"huggingface"] = metadata.get(b"huggingface", b"")

    schema = pa.schema(
        [
            pa.field(name, source_field.type if name == ACTION_TARGET_KEY else table.schema.field(name).type)
            for name in names
        ],
        metadata=table.schema.metadata,
    )
    return pa.Table.from_arrays(columns, schema=schema)


def transform_parquet_files(source: Path, dest: Path) -> None:
    source_data = source / "data"
    dest_data = dest / "data"
    for parquet_path in sorted(source_data.rglob("*.parquet")):
        rel = parquet_path.relative_to(source)
        dest_path = dest / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(parquet_path)
        transformed = transform_table(table)
        pq.write_table(transformed, dest_path)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()

    validate_source(source)

    if dest.exists():
        if not args.force:
            raise FileExistsError(f"Destination already exists: {dest}")
        shutil.rmtree(dest)

    copy_tree_with_symlinks(source, dest)
    copy_meta_extras(source, dest)
    update_info_json(source, dest)
    update_stats_json(source, dest)
    transform_parquet_files(source, dest)

    print(f"Prepared pi05 dataset at: {dest}")
    print(f"Use dataset.root={dest}")


if __name__ == "__main__":
    main()
