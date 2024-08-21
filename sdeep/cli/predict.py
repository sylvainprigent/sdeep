"""CLI to run a prediction where input and output are images"""
from pathlib import Path
import argparse

from sdeep.api import SDeepAPI


def main():
    """CLI main function"""
    parser = argparse.ArgumentParser(description='SDeep predict image '
                                                 'restoration')
    parser.add_argument('-i', '--input', help='Input data file or dir', default='')
    parser.add_argument('-m', '--model', help='Model file', default='')
    parser.add_argument('-o', '--output', help='Output file or dir', default='')
    parser.add_argument('-e', '--ext', help='Output file extension (only when input is '
                                            'a directory)', default='')
    parser.add_argument('-b', '--batch', help='Size of processing batch (only when '
                                              'input is a directory)', default=1)
    parser.add_argument('-p', '--tiling', help='True to use tiling, False  otherwise',
                        default='false')

    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)

    # check inputs
    if not input_path.exists():
        print('Error: the input image file does not exist')
        return
    if not model_path.exists():
        print('Error: the model file does not exist')
        return

    tiling = False
    if args.tiling == "true":
        tiling = True

    api = SDeepAPI()
    api.predict(model_path, input_path, output_path,
                out_extension=args.ext,
                batch_size=int(args.batch),
                use_tiling=tiling)


if __name__ == "__main__":
    main()
