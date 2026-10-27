"""Generate GenEval images using LightX2V's synchronous HTTP API."""

import argparse
import io
import json
from pathlib import Path
import sys

from PIL import Image
import requests


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_file", type=Path)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--n-samples", "--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Default seed when metadata has no seed")
    parser.add_argument("--size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.n_samples <= 0 or args.seed < 0 or args.timeout <= 0:
        parser.error("samples/timeout must be positive; seed must be non-negative")
    if args.size and min(args.size) <= 0:
        parser.error("height and width must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    with args.metadata_file.open(encoding="utf-8") as stream:
        metadatas = [json.loads(line) for line in stream]
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise ValueError("Use a new or empty --outdir for each run")
    endpoint = args.server_url.rstrip("/") + "/v1/tasks/image/sync"

    with requests.Session() as session:
        for index, metadata in enumerate(metadatas):
            folder = args.outdir / f"{index:05d}"
            samples = folder / "samples"
            samples.mkdir(parents=True, exist_ok=True)
            (folder / "metadata.jsonl").write_text(
                json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            for sample in range(args.n_samples):
                payload = {
                    "task": "t2i", "prompt": metadata["prompt"],
                    "seed": metadata.get("seed", args.seed) + sample,
                }
                if args.size is not None:
                    payload["size"] = args.size  # LightX2V expects [height, width].
                if args.negative_prompt is not None:
                    payload["negative_prompt"] = args.negative_prompt
                response = session.post(
                    endpoint, json=payload,
                    params={"timeout_seconds": args.timeout},
                    timeout=(10, args.timeout + 30),
                )
                response.raise_for_status()
                if response.headers.get("Content-Type", "").split(";")[0] != "image/png":
                    raise ValueError("Server did not return image/png")
                with Image.open(io.BytesIO(response.content)) as image:
                    if image.format != "PNG":
                        raise ValueError("Server returned an invalid PNG")
                    if args.size and image.size != (args.size[1], args.size[0]):
                        raise ValueError(f"Unexpected image size: {image.size}")
                    image.verify()
                with Image.open(io.BytesIO(response.content)) as image:
                    image.load()
                output = samples / f"{sample:05d}.png"
                temporary = samples / f".{sample:05d}.png.tmp"
                temporary.write_bytes(response.content)
                temporary.replace(output)
                print(f"[{index + 1}/{len(metadatas)}] saved {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
