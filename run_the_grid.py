import bz2
import lzma
import gzip
import json
import pathlib
import argparse
import logging
import functools
from datetime import datetime
from urllib.parse import urlparse
import shutil
import multiprocessing
import socket

import executor
import requests
import gspread


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


def get_worksheet(api_key_path, spreadshet_id, worksheet_name="Params combined"):
    try:
        gc = gspread.service_account(filename=api_key_path)
        sh = gc.open_by_key(spreadshet_id)
        logger.info(f"Successfully opened spreadshet {sh.title}")
    except gspread.exceptions.APIError:
        logger.error(
            f"Cannot connect to google spreadsheet, check the validity of your service account api key {api_key_path} or spreadshet id {spreadshet_id}"
        )
        return
    except gspread.exceptions.SpreadsheetNotFound:
        logger.error(f"Cannot find google spreadsheet, check your spreadshet id {spreadshet_id}")
        return

    try:
        return sh.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        logger.error(f"Cannot open worksheet {worksheet_name} in {spreadsheet.title}, please fix the spreadsheet")
        return


def pick_the_task(worksheet, hostname):
    tasks = worksheet.get_all_records()

    for row_no, t in enumerate(tasks):
        if t["Status"] == "":
            logger.info(f"We've found a new task at row {row_no + 2}, it is {t['Description']}")
            try:
                algo, epochs, subwords, wordngram, neg_sampling = t["Params"].strip().split(";")
                subwords_min, subwords_max = subwords.split("-")
            except ValueError:
                logger.error(f"Cannot parse record {t['Params']}, skipping it")
                continue

            worksheet.update_cell(row_no + 2, 3, "Processing")
            worksheet.update_cell(row_no + 2, 4, str(datetime.now()))
            worksheet.update_cell(row_no + 2, 5, hostname)

            return {
                "task": t,
                "params": {
                    "algo": algo,
                    "epochs": epochs,
                    "subwords_min": subwords_min,
                    "subwords_max": subwords_max,
                    "wordngram": wordngram,
                    "neg_sampling": neg_sampling,
                },
                "suffix": f"algo-{algo}.epochs-{epochs}.subwords-{subwords_min}..{subwords_max}.wordngram-{wordngram}.neg_sampling-{neg_sampling}",
                "row_no": row_no + 2,
            }

    return

def tick_the_task(worksheet, task):
    worksheet.update_cell(task["row_no"], 3, "Computed")
    worksheet.update_cell(task["row_no"], 4, str(datetime.now()))


def train(args):
    logger.info("Running pre-flight checks")
    if not args.config.exists():
        logger.error(f"Config file {args.config} doesn't exist, please run ./run_the_grid.py setup first")
        return

    try:
        with open(args.config, "r") as fp:
            config = json.load(fp)
    except json.decoder.JSONDecodeError:
        logger.error(
            f"Config file {args.config} is corrupt, please run ./run_the_grid.py setup again and do not tamper with file"
        )
        return

    corpus_path = pathlib.Path(config["corpus"])
    if not corpus_path.exists():
        logger.error(f"Corpus file {corpus_path} doesn't exist, please re-run ./run_the_grid.py setup")
        return

    vectors_path = pathlib.Path(config["vectors"])
    if not vectors_path.exists():
        logger.error(f"Vectors directory {vectors_path} doesn't exist, please re-run ./run_the_grid.py setup")
        return

    api_key_path = pathlib.Path(config["api_key"])
    if not api_key_path.exists():
        logger.error(f"Api key file {api_key_path} doesn't exist, please re-run ./run_the_grid.py setup")
        return

    fasttext_path = pathlib.Path(config["fasttext"])
    if not fasttext_path.exists() or not executor.is_executable(fasttext_path):
        logger.error(
            f"Fastext binary {fasttext_path} doesn't exist or it is not executable, please re-run ./run_the_grid.py setup"
        )
        return

    worksheet = get_worksheet(api_key_path, config.get("spreadshet_id"))
    if worksheet is None:
        return

    while True:
        task = pick_the_task(worksheet, config.get("hostname", socket.gethostname()))
        if task is None:
            logger.warning(f"Woohoo, no more tasks left in the queue")
            return

        vectors_fname = vectors_path / (corpus_path.name + "." + task["suffix"])

        executor.execute(
            fasttext_path,
            task["params"]["algo"],
            "-epoch",
            task["params"]["epochs"],
            "-neg",
            task["params"]["neg_sampling"],
            "-wordNgrams",
            task["params"]["wordngram"],
            "-minn",
            task["params"]["subwords_min"],
            "-maxn",
            task["params"]["subwords_max"],
            "-input",
            corpus_path,
            "-output",
            vectors_fname,
            "-dim",
            "300",
            "-threads",
            config["threads"],
        )
        
        vectors_plaintext = vectors_fname.parent / (vectors_fname.name + '.vec')
        if vectors_plaintext.exists():
            vectors_plaintext.unlink()
        vectors_bin = vectors_fname.parent / (vectors_fname.name + '.bin')

        with open(config["logfile"], "a") as fp_out:
            fp_out.write(
                json.dumps({
                    "vectors": str(vectors_bin),
                    "corpus": str(corpus_path),
                    "params": task["params"],
                    "dt": str(datetime.now()),
                }, sort_keys=True) + "\n"
            )
        
        tick_the_task(worksheet, task)

def setup(args):
    # Checking dependencies
    binary_dependencies = ["git", "make"]
    for dep in binary_dependencies:
        if shutil.which(dep) is None:
            logger.error(f"Cannot find required binary dependencies {dep}, exiting. Please install it and return back")
            return

    # Checking api keys
    if not args.api_key_location.exists():
        logger.error(f"Api key file {args.api_key_location} doesn't exist")
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

    # TODO: verify the case when it's already exists
    decompressed_corpus_path = corpus_path

    if not corpus_path.exists() or args.overwrite_corpus:
        logger.info(f"Downloading vectors from {args.corpus_url} to {corpus_path}")
        download_file(args.corpus_url, corpus_path)
        logger.info(f"Corpus successfully downloaded to {corpus_path}")

    corpus_fh = open_compressed(corpus_path)

    if corpus_fh is not None:
        decompressed_corpus_path = corpus_path.with_suffix("")

        if not decompressed_corpus_path.exists() or args.overwrite_corpus:
            logger.info(f"Decompressing {corpus_path.suffix} corpus")

            with open(decompressed_corpus_path, "w") as fp_out:
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

        if not executor.execute(
            "git", "clone", "https://github.com/facebookresearch/fastText.git", args.fasttext_location, check=False
        ):
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
            json.dump(
                {
                    "corpus": str(decompressed_corpus_path),
                    "fasttext": str(fasttext_path),
                    "api_key": str(args.api_key_location),
                    "spreadshet_id": args.spreadshet_id,
                    "threads": args.threads,
                    "vectors": str(args.vectors_location),
                    "logfile": str(args.logfile),
                    "hostname": args.hostname,
                },
                fp_out,
                sort_keys=True,
                indent=4,
            )

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

    # Setup subparser
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

    setup_parser.add_argument(
        "--api_key_location",
        type=pathlib.Path,
        help="Location of json file with service account credentials for google drive and spreadsheet",
        default=pathlib.Path("api_keys/fasttext_gridtraining.json"),
    )

    setup_parser.add_argument(
        "--spreadshet_id",
        type=str,
        help="Google Spreadsheet id (the one from the url) with the spreadsheet of tasks and results",
        default="150DjEZKCuJEcsCJWahWmhPkfHzn9pA-N3UIYYx7XM04",
    )

    setup_parser.add_argument(
        "--threads", type=int, help="Number of threads to use", default=multiprocessing.cpu_count() - 2
    )

    setup_parser.add_argument(
        "--logfile", type=pathlib.Path, help="JSONLines file to write training details",
        default=pathlib.Path("log.jsonl")
    )

    setup_parser.add_argument(
        "--hostname", type=str, help="Identifier of the worker, defaulted to the hostname",
        default=socket.gethostname()
    )

    setup_parser.set_defaults(func=setup)

    # Train subparser
    train_parser = subparsers.add_parser("train", help="Run the trainings")
    train_parser.set_defaults(func=train)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    args.func(args)
