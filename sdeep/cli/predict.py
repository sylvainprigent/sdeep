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
    parser.add_argument('-i', '--input', help='Input data file', default='')
    parser.add_argument('-m', '--model', help='Model file', default='')
    parser.add_argument('-t', '--transform', help='Input data transformation', default='')
    parser.add_argument('-o', '--output', help='Output file', default='')
    parser.add_argument('-p', '--tiling', help='True to use tiling, False '
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

    # apply transform
    api = SDeepAPI()
    transform = api.load_model(args.transform)
    image = transform(image)

    model = api.load_model(args.model)
    prediction = api.predict(model, image, args.tiling)

    # save the image
    if prediction.ndim == 1:
        if prediction.shape[0] == 1:
            with open(args.output, 'w', encoding='utf-8') as file:
                file.write(f"{prediction.item()}")
        else:
            np.savetxt(args.output, prediction, delimiter=",")
    else:
        imsave(args.output, prediction)


if __name__ == "__main__":
    main()
