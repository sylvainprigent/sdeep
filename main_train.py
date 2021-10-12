import torch
from torch.utils.data import DataLoader
from sdeep.models import DnCNN
from sdeep.workflows import SWorkflow
from sdeep.datasets import RestorationPatchDataset, RestorationDataset


if __name__ == "__main__":
    model = DnCNN()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_source_dir = '/home/sprigent/Documents/datasets/airbus/original_250_small/train/noise120/'
    train_target_dir = '/home/sprigent/Documents/datasets/airbus/original_250_small/train/GT'
    train_dataset = RestorationPatchDataset(train_source_dir,
                                            train_target_dir,
                                            patch_size=40,
                                            stride=10,
                                            use_data_augmentation=True)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=128,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=0)

    test_source_dir = '/home/sprigent/Documents/datasets/airbus/original_250_small/test/noise120/'
    test_target_dir = '/home/sprigent/Documents/datasets/airbus/original_250_small/test/GT/'
    test_dataset = RestorationDataset(test_source_dir,
                                      test_target_dir,
                                      use_data_augmentation=False)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=3,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=0)

    scheme = SWorkflow(model, loss_fn, optimizer,
                       train_data_loader,
                       test_data_loader,
                       epochs=50)
    scheme.fit()
