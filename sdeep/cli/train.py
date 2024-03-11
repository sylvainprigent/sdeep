"""Entry point for train CLI"""
import os
import argparse

from pathlib import Path

from sdeep.api import SDeepAPI
from sdeep.utils import SParametersReader


def get_subdir(main_dir: str):
    """Create the output directory path

    :param main_dir: parent run directory
    :return: newly created run directory
    """
    run_id = 1
    path = os.path.join(main_dir, f"run_{run_id}")
    while os.path.isdir(path):
        run_id += 1
        path = os.path.join(main_dir, f"run_{run_id}")
    Path(path).mkdir(parents=True)
    return path


def main():
    """Main function for train CLI"""
    parser = argparse.ArgumentParser(description='SDeep train',
                                     conflict_handler='resolve')
    parser.add_argument('-p', '--parameters',
                        help='Parameter file', default='')
    parser.add_argument('-s', '--save',
                        help='Save directory', default='./runs')
    parser.add_argument('-r', '--reuse',
                        help='True to reuse a previous checking point',
                        default='false')

    args = parser.parse_args()

    if args.reuse == "true":
        out_dir = args.save
    else:
        out_dir = get_subdir(args.save)

    params = SParametersReader.read(Path(args.parameters))
    SParametersReader.write(params, Path(out_dir) / "params.json")
    SDeepAPI().train(params, Path(out_dir))


if __name__ == "__main__":
    main()
