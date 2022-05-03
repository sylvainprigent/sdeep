import torch
from skimage import draw


class FRCLoss(torch.nn.Module):
    """Define an image reconstruction loss with the Fourier Ring Correlation

    Parameters
    ----------
    patch_size: int
        size of the input image patch in it smallest dimension

    """
    def __init__(self, patch_size):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights = torch.ones()
        self.linear = torch.nn.Linear(len(self.weights), 1)
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(
                torch.Tensor(self.weights).unsqueeze(0).unsqueeze(0).to(self.device))

    def forward(self, input, target):
        # calculate fourier transform
        input_fft = torch.fft.fft2(input)
        target_fft = torch.fft.fft2(target)

        # calculate the correlation for each ring
        s_x = input.shape[2]
        s_y = input.shape[3]
        r_max = len(self.weights)  # min(int(s_x / 2), int(s_y / 2))
        curve_ = torch.zeros((input.shape[0], input.shape[1], r_max - 1))
        curve_[:, :, 0] = 1
        for radius in range(1, r_max - 1):
            r_r, c_c = draw.circle_perimeter(int(s_x / 2), int(s_y / 2), radius)
            p_1 = input_fft[:, :, r_r, c_c]
            p_2 = target_fft[:, :, r_r, c_c]

            num = torch.abs(torch.sum(p_1 * torch.conj(p_2)))
            den = torch.sum(torch.square(torch.abs(p_1))) * torch.sum(torch.square(torch.abs(p_2)))

            curve_[radius] = num / torch.sqrt(den)

        # loss is linear combination of ring correlation
        return self.linear(curve_)
