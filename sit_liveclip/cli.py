import logging
from pathlib import Path
from argparse import ArgumentParser

from sit_liveclip import experiment, __version__
from sit_liveclip.config import Config
from sit_liveclip.core.env import LiveClipEnv, DummyLiveClipEnv
from sit_liveclip.core.player_connection import PlayerConnection


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=f"{args.command}.log", level=logging.DEBUG)
    logging.getLogger("numba").setLevel(logging.WARNING)

    config = Config()

    env = (
        DummyLiveClipEnv(
            config.history_size,
            config.sliding_window_size,
            config.segment_size,
        )
        if args.env == "dummy"
        else LiveClipEnv(
            player_conn=PlayerConnection(base_url=config.player_conn_base_url),
            history_size=config.history_size,
            sliding_window_size=config.sliding_window_size,
            segment_size=config.segment_size,
        )
    )

    if args.command == "train":
        experiment.train(env, args.logdir, config)
    elif args.command == "eval":
        experiment.evaluate(env, args.model, config)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="sit_liveclip", description="Run experiment")

    parser.add_argument(
        "-e",
        "--env",
        help="Environment ['dummy', 'live']",
        choices=["dummy", "live"],
        dest="env",
        metavar="ENV",
        required=True,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print version info",
    )

    subparsers = parser.add_subparsers(dest="command")

    parser_train = subparsers.add_parser("train", help="Training mode")
    parser_train.add_argument(
        "--logdir",
        help="Log dir",
        metavar="LOG_DIR",
        default=None,
        type=Path,
    )

    parser_eval = subparsers.add_parser("eval", help="Evaluation mode")
    parser_eval.add_argument(
        "--model",
        help="Saved model path",
        metavar="MODEL_PATH",
        required=True,
        type=Path,
    )

    return parser
