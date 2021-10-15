import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.io import imread
from sdeep.models import DnCNN



model_filename = "saved_models/dncnn8_original_250_small.pt"

# load
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DnCNN(num_of_layers=8, channels=1, features=64)
model.load_state_dict(torch.load(model_filename))
model.to(device)

# predict
gt = 2000*np.ones((512, 512))
gt[128:184, 128:184] = 100
x = 0.046*np.random.poisson(gt/0.046) + np.random.normal(0, 7.7)

#gt = np.float32(imread('/home/sprigent/Documents/datasets/airbus/original/test/GT/Brest0146_6836_50cm_F06opt.tif'))
#x = np.float32(imread('/home/sprigent/Documents/datasets/airbus/original/test/noise120/Brest0146_6836_50cm_F06opt.tif'))
x = np.float32(imread('/home/sprigent/Documents/codes/sdeep/IMG0032_sted.tif'))

x_torch = torch.from_numpy(x).view(1, 1, * x.shape).float()
x_device = x_torch.to(device)
model.eval()
with torch.no_grad():
    pred = model(x_device)
pred_numpy = pred[0, 0, :, :].cpu().numpy()


#fig0 = plt.figure()
#plt.imshow(gt, cmap='gray')

fig1 = plt.figure()
plt.imshow(x, cmap='gray')

fig2 = plt.figure()
plt.imshow(pred_numpy, cmap='gray')

plt.show()
