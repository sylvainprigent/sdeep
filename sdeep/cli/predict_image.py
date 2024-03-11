"""CLI to run a prediction where input and output are images"""
import os
import argparse

import numpy as np
from skimage.io import imread, imsave

from sdeep.api import SDeepAPI


def main():
    """CLI main function"""
    parser = argparse.ArgumentParser(description='SDeep predict image '
                                                 'restoration')
    parser.add_argument('-i', '--input', help='Input image file', default='')
    parser.add_argument('-m', '--model', help='Model file', default='')
    parser.add_argument('-o', '--output', help='Output image file', default='')
    parser.add_argument('-t', '--tiling', help='True to use tiling, False '
                                               'otherwise',
                        default='False')
    args = parser.parse_args()

    # check inputs
    if not os.path.isfile(args.input):
        print('The input image file does not exist')
        return
    if not os.path.isfile(args.model):
        print('The model file does not exist')
        return

    # load input image
    image = np.float32(imread(args.input))

    api = SDeepAPI()
    model = api.load_model(args.model)
    pred_image = api.predict(model, image, args.tiling)

    # save the image
    imsave(args.output, pred_image)


if __name__ == "__main__":
    main()
