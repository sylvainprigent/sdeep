import os
import numpy as np
import torch
import argparse
from skimage.io import imread, imsave

from sdeep.utils import TilePredict
from sdeep.factories import sdeepModels


def main():
    parser = argparse.ArgumentParser(description='SDeep predict image restoration')

    parser.add_argument('-i', '--input', help='Input image file', default='')
    parser.add_argument('-m', '--model', help='Model file', default='')
    parser.add_argument('-o', '--output', help='Output image file', default='')
    parser.add_argument('-t', '--tiling', help='True to use tiling, False otherwise',
                        default='False')
    args = parser.parse_args()

    # check inputs
    if not os.path.isfile(args.input):
        print('The input image file does not exist')
        return
    if not os.path.isfile(args.model):
        print('The model file does not exist')
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device = ', device)

    # read the image
    image = np.float32(imread(args.input))
    image_torch = torch.from_numpy(image).float()
    image_device = image_torch.to(device).unsqueeze(0).unsqueeze(0)

    # read the model
    check_point = torch.load(args.model, map_location=torch.device(device))
    model = sdeepModels.get_instance(check_point['model'], check_point['model_args'])
    model.load_state_dict(check_point['model_state_dict'])
    model.to(device)
    model.eval()

    # run the model
    if args.tiling:
        tile_predict = TilePredict(model)
        pred = tile_predict.run(image_device)
    else:
        with torch.no_grad():
            pred = model(image_device)

    # save the image
    pred_numpy = pred[0, 0, :, :].cpu().numpy()
    imsave(args.output, pred_numpy)


if __name__ == "__main__":
    main()
