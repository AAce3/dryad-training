import argparse
import shlex
from argparse import ArgumentParser
from data_processing import auto_convert, process_leela_data


def auto_download(
    output_dir: str,
    rescorer_path: str,
    syzygy_paths: str,
    max_files: int,
    rescorer_args: str = str,
    num_threads: int = 1,
    sample_rate: int = 32,
    uncertainty_lambda: float = 5.0 / 6.0,
    excluded_files_path: str = "",
):
    rescore_args_split = shlex.split(rescorer_args)
    auto_convert(
        output_dir,
        rescorer_path,
        syzygy_paths,
        rescore_args_split,
        max_files,
        num_threads,
        sample_rate,
        uncertainty_lambda,
        excluded_files_path,
    )


def process_data(
    input_dir: str,
    output_dir: str,
    rescorer_path: str = "",
    syzygy_paths: str = "",
    rescorer_args: str = str,
    sample_rate: int = 32,
    uncertainty_lambda: float = 5.0 / 6.0,
    delete_original: bool = False,
):
    rescore_args_split = shlex.split(rescorer_args)
    process_leela_data(
        input_dir,
        output_dir,
        rescorer_path,
        syzygy_paths,
        rescore_args_split,
        sample_rate,
        uncertainty_lambda,
        delete_original,
    )


SUBCMD_NAME = "dataprocess_subcommand"


def add_args(parser: ArgumentParser):
    traindata_parser = parser.add_subparsers(title="Available modes", dest=SUBCMD_NAME)

    auto_download_parser: ArgumentParser = traindata_parser.add_parser(
        "auto-process",
        help="Automatically download, rescore and preprocess training data from lc0",
    )
    auto_download_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for preprocessed files",
    )
    auto_download_parser.add_argument(
        "--max-files",
        type=int,
        required=True,
        help="Maximum number of .tar files to download from lc0 website. Each .tar file contains >10,000 games",
    )
    auto_download_parser.add_argument(
        "--rescorer-path",
        type=str,
        required=True,
        help="Path to the lc0 rescorer to preprocess data",
    )
    auto_download_parser.add_argument(
        "--syzygy-paths",
        type=str,
        required=True,
        help="Path to Syzygy tablebases (required by lc0 rescorer)",
    )
    auto_download_parser.add_argument(
        "--rescorer-args",
        type=str,
        default="",
        help='Optional additional arguments for lc0 rescorer. You can find these using "./rescorer rescore --help"',
    )
    auto_download_parser.add_argument(
        "--num-threads",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of threads used to make requests. If you have enough network bandwidth, this will lead to significant speedups.",
    )
    auto_download_parser.add_argument(
        "--excluded_files_path",
        type=str,
        default=argparse.SUPPRESS,
        help="Exclude certain files from being downloaded. This is useful when downloading data throughout multiple sessions, as the program will automatically list out which files were downloaded during runtime",
    )

    direct_process_parser: ArgumentParser = traindata_parser.add_parser(
        "process-existing", help="Process existing data"
    )
    direct_process_parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory for preprocessed files",
    )
    direct_process_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for preprocessed files",
    )
    direct_process_parser.add_argument(
        "--rescorer-path",
        type=str,
        default=argparse.SUPPRESS,
        help="Optional path to the lc0 rescorer to preprocess data",
    )
    direct_process_parser.add_argument(
        "--syzygy-paths",
        type=str,
        default=argparse.SUPPRESS,
        help="Optional path to Syzygy tablebases (required if lc0 rescorer is being used)",
    )
    direct_process_parser.add_argument(
        "--rescorer-args",
        type=str,
        default="",
        help='Optional additional arguments for lc0 rescorer. You can find these using "./rescorer rescore --help"',
    )
    direct_process_parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Whether or not to delete the original files",
    )


def process_traindata_args(args: dict):
    subcommand = args.pop(SUBCMD_NAME)

    if subcommand == "auto-process":
        auto_download(**args)
    elif subcommand == "process-existing":
        process_data(**args)
    else:
        raise "Invalid arguments!"


def main():
    parser = argparse.ArgumentParser(prog="Testing")
    add_args(parser)
    args = vars(parser.parse_args())
    process_traindata_args(args)


if __name__ == "__main__":
    main()
