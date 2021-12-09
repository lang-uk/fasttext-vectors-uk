import bz2
import lzma
import gzip
import json
import pathlib
import argparse
import logging
import functools
from urllib.parse import urlparse
import shutil

import executor
import requests


logger = logging.getLogger("run_the_grid")


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raw.read = functools.partial(r.raw.read, decode_content=True)
        r.raise_for_status()

        with open(local_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)


def open_compressed(filename):
    if filename.suffix == ".bz2":
        return bz2.open(filename, "rt")

    if filename.suffix == ".gz":
        return gzip.open(filename, "rt")

    if filename.suffix == ".xz":
        return lzma.open(filename, "rt")

    return None


def setup(args):
    # Checking dependencies
    binary_dependencies = ["git", "make"]
    for dep in binary_dependencies:
        if shutil.which(dep) is None:
            logger.error(f"Cannot find required binary dependencies {dep}, exiting. Please install it and return back")
            return

    # Creating folders
    try:
        # logger.info(f"Creating lib folder {args.fasttext_location}")
        # args.fasttext_location.mkdir(exist_ok=True)
        logger.info(f"Creating corpus folder {args.corpus_location}")
        args.corpus_location.mkdir(exist_ok=True)
        logger.info(f"Creating vectors folder {args.vectors_location}")
        args.vectors_location.mkdir(exist_ok=True)
    except (PermissionError, FileNotFoundError) as e:
        logger.error(f"Cannot create one of the required folders: {e}")
        return

    # Downloading and unpacking corpus
    corpus_frags = urlparse(args.corpus_url)
    corpus_path = args.corpus_location / pathlib.Path(corpus_frags.path).name

    if not corpus_path.exists() or args.overwrite_corpus:
        logger.info(f"Downloading vectors from {args.corpus_url} to {corpus_path}")
        download_file(args.corpus_url, corpus_path)
        logger.info(f"Corpus successfully downloaded to {corpus_path}")

        corpus_fh = open_compressed(corpus_path)

        if corpus_fh is not None:
            logger.info(f"Decompressing {corpus_path.suffix} corpus")

            with open(corpus_path.with_suffix(""), "w") as fp_out:
                shutil.copyfileobj(corpus_fh, fp_out)

            corpus_fh.close()

            logger.info(f"Finished decompressing {corpus_path.suffix} corpus")
    else:
        logger.info(f"Corpus file {corpus_path} already exists, skipping")

    fasttext_path = args.fasttext_location / "fasttext"

    # building fasttext
    if not fasttext_path.exists() or args.overwrite_fasttext:
        if args.fasttext_location.exists():
            shutil.rmtree(args.fasttext_location)

        logger.info("Cloning fasttext repo")

        if not executor.execute("git", "clone", "https://github.com/facebookresearch/fastText.git", args.fasttext_location, check=False):
            logger.error("Failed to clone fasttext repo")
            return

        logger.info("Building fasttext")
        if not executor.execute("make", directory=args.fasttext_location, check=False):
            logger.error("Failed to build fasttext binaries")
            return

        logger.info(f"Fasttext binaries successfully cloned and built at {fasttext_path}")
    else:
        logger.info(f"Fasttext binary {fasttext_path} already exists, skipping")

    if not args.config.exists() or args.overwrite_config:
        with open(args.config, "w") as fp_out:
            json.dump({
                "corpus": str(corpus_path),
                "fasttext": str(fasttext_path),
                "more": "to come",
            }, fp_out, sort_keys=True, indent=4)

        logger.info(f"Config stored to {args.config}")
    else:
        logger.warning(f"Config {args.config} already exists, not overwriting")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""That is the node worker to compute fasttext """
        """vectors using different params, store obtained vectors on gdrive and update google spreadsheet"""
    )

    parser.add_argument("--config", type=pathlib.Path, help="Path to config file", default="config.json")
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    subparsers = parser.add_subparsers(help="Available commands")
    setup_parser = subparsers.add_parser(
        "setup", help="Download and build dependencies, download corpus file, create config from template"
    )

    setup_parser.add_argument(
        "--overwrite_config",
        help="Overwrite config if it is already exists",
        action="store_true",
        default=False,
    )

    setup_parser.add_argument(
        "--overwrite_corpus",
        help="Overwrite corpus if it is already exists",
        action="store_true",
        default=False,
    )

    setup_parser.add_argument(
        "--overwrite_fasttext",
        help="Download and rebuild fasttext, if it is already exists",
        action="store_true",
        default=False,
    )

    setup_parser.add_argument(
        "--corpus_location",
        type=pathlib.Path,
        help="Download corpus to specific folder",
        default=pathlib.Path("corpus"),
    )

    setup_parser.add_argument(
        "--corpus_url",
        type=str,
        help="Download corpus to specific folder",
        default="https://lang-uk.nbu.rocks/static/ubertext.fiction_news_wikipedia.filter_rus+short.tokens.txt.bz2",
    )

    setup_parser.add_argument(
        "--fasttext_location",
        type=pathlib.Path,
        help="Download and build fasttext to specific folder",
        default=pathlib.Path("lib"),
    )

    setup_parser.add_argument(
        "--vectors_location",
        type=pathlib.Path,
        help="Store vectors to given folder after the training",
        default=pathlib.Path("vectors"),
    )
    setup_parser.set_defaults(func=setup)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    args.func(args)
