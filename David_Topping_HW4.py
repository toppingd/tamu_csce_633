import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import numpy as np
import glob
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from time import time

import torchvision

class SUN397Dataset(Dataset):
    """
    A custom dataset class for loading the SUN397 dataset.
    """

    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset with images and labels.
        Args:
            data_dir (str): Path to the data directory.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_dir = data_dir
        self.transform = transform

        self.means = [0.5285, 0.4667, 0.4108]
        self.stds = [0.2243, 0.2312, 0.2345]

        self.transform = transforms.Compose([transforms.Resize((224,224)),
                                              transforms.ToTensor(),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.Normalize(mean=self.means,
                                                                   std=self.stds)])

        self.dataset = torchvision.datasets.ImageFolder(root=self.data_dir,
                                                        transform=self.transform)


    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.dataset)


    def __getitem__(self, idx):
        """
        Retrieves an image and its label at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (image, label)
        """

        return self.dataset[idx][0], self.dataset[idx][1]


class CNN(nn.Module):
    """
    Define your CNN Model here
    """
    def __init__(self, num_classes=4):
        """
        Initializes the layers of the CNN model.

        Args:
            num_classes (int): Number of output classes.
        """
        super(CNN, self).__init__()

        # Convolutional layers
        self.layers = nn.Sequential(

            nn.Conv2d(in_channels = 3, out_channels = 64,
                      kernel_size = 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels = 64, out_channels = 128,
                      kernel_size = 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(p=0.2),

            )

        # fully connected layers
        self.fc = nn.Sequential(

            nn.Linear(in_features=56 * 56 * 128, out_features=120),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(in_features=120, out_features=60),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(in_features=60, out_features=num_classes)

            )


    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output of the model.
        """

        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor

        return self.fc(x)


def calculate_mean_std(**kwargs):
    """
    Fill in the per channel mean and standard deviation of the dataset.
    Just fill in the values, no need to compute them.
    """

    # transform = transforms.ToTensor()
    # dataset = torchvision.datasets.ImageFolder(root=kwargs['data_dir'],
    #                                            transform=transform)

    # dataloader = DataLoader(dataset, batch_size=32, num_workers=24)
    # means = torch.zeros(3)
    # stds = torch.zeros(3)
    # total_images = 0
    # for images, _ in dataloader:
    #     batch_size = images.size(0)
    #     total_images += batch_size
    #     for i in range(len(means)):
    #         means[i] += images[:,i,:,:].mean() * batch_size
    #         stds[i] += images[:,i,:,:].std() * batch_size

    # means /= total_images
    # stds /= total_images

    # mean_r, mean_g, mean_b = [np.round(mean.item(),4) for mean in means]
    # std_r, std_g, std_b = [np.round(std.item(),4) for std in stds]

    #mean_r, mean_g, mean_b = (0.5286, 0.4668, 0.4108)
    #std_r, std_g, std_b = (0.2243, 0.2319, 0.2345)

    # data_dir = kwargs['data_dir']

    # means = np.zeros(3)
    # stds = np.zeros(3)
    # total_pixels = 0

    # count = 0
    # for subdir in os.listdir(data_dir):
    #     subdir_path = os.path.join(data_dir, subdir)
    #     if os.path.isdir(subdir_path):
    #         for nested_subdir in os.listdir(subdir_path):
    #             nested_subdir_path = os.path.join(subdir_path, nested_subdir)
    #             if os.path.isdir(nested_subdir_path):
    #                 for filename in os.listdir(nested_subdir_path):
    #                     if filename.endswith('.jpg'):
    #                         image = Image.open(os.path.join(nested_subdir_path, filename))
    #                         image_np = np.array(image)

    #                         if len(image_np.shape) == 2: # Grayscale image
    #                             image_np = np.stack([image_np] * 3, axis=-1) # Convert to RGB by duplicating channels

    #                         total_pixels += image_np.shape[0] * image_np.shape[1]

    #                         for i in range(3):
    #                             means[i] += np.mean(image_np[:, :, i]) * image_np.shape[0] * image_np.shape[1]
    #                             stds[i] += np.std(image_np[:, :, i]) * image_np.shape[0] * image_np.shape[1]

    #                         count+=1
    #                         print(count, os.path.join(nested_subdir_path, filename))

    # means /= total_pixels
    # stds /= total_pixels

    # mean_r, mean_g, mean_b = [np.round(mean, 4) for mean in means]
    # std_r, std_g, std_b = [np.round(std, 4) for std in stds]

    mean_r, mean_g, mean_b = (0.5285, 0.4667, 0.4108)
    std_r, std_g, std_b = (0.2243, 0.2312, 0.2345)

    # mean_r, mean_g, mean_b = (130.7403, 115.4837, 101.348)
    # std_r, std_g, std_b = (56.0456, 57.9080, 58.5345)

    return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]


'''
All of the following functions are optional. They are provided to help you get started.
'''

def train(model, train_loader, test_loader, **kwargs):

    device = kwargs['device']
    loss_fn = kwargs['crit']
    optimizer = kwargs['optimizer']
    scheduler = kwargs['scheduler']
    n_epochs = kwargs['num_epochs']

    train_loss = []
    test_loss = []
    best_loss = float('inf')

    with Progress(
        TextColumn("[bold blue]Training Epoch {task.fields[epoch]}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:

        for epoch in range(n_epochs):

            model.train()

            epoch_train_loss = []

            task = progress.add_task("train", total=len(train_loader), epoch=epoch)

            for batch, (X, y) in enumerate(train_loader):

                X, y = X.to(device), y.to(device)
                preds = model(X)

                loss = loss_fn(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss.append(loss.item())

                progress.update(task, advance=1)
                if batch % 100 == 0:
                    print(f'Train Epoch: {epoch} [{batch * len(X)}/{len(train_loader.dataset)} '
                          f'({100. * batch / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            train_loss.append(np.mean(epoch_train_loss))

            scheduler.step()

            # Evaluate on test set
            model.eval()

            epoch_test_loss = []

            with torch.no_grad():
                for batch, (X, y) in enumerate(test_loader):
                    X, y = X.to(device), y.to(device)
                    preds = model(X)
                    loss = loss_fn(preds, y)
                    epoch_test_loss.append(loss.item())

            test_loss.append(np.mean(epoch_test_loss))

            print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}')

            # Save the best model
            if test_loss[-1] < best_loss:
                best_loss = test_loss[-1]
                torch.save(model.state_dict(), 'model4.pt')
                print(f'Best model saved with test loss: {best_loss:.4f}')


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_dir', type=str,
#                         default='welcome/to/CNN/homework',
#                         help='Path to training data directory')

#     parser.add_argument('--seed', type=int, default=42,
#                         help='Random seed for reproducibility')

#     return parser.parse_args()


def main():
    # args = parse_args()
    # torch.manual_seed(args.seed)

    # data_dir = 'C:\\Users\\dtopp\\Documents\\data'
    data_dir = '/scratch/user/dtopping/csce_633/hw4/data'

    # global means, stds
    # means, stds = calculate_mean_std(data_dir=data_dir)
    # print(f'Means: {means}, Stds: {stds}')

    ds = SUN397Dataset(data_dir)

    # create training and testing datasets
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=24)
    test_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=24)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the CNN
    model = CNN()

    # Loss function
    crit = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train model
    train(model, train_loader, test_loader, device=device, crit=crit,
          optimizer=optimizer, scheduler=scheduler, num_epochs=30)


if __name__ == "__main__":
    main()
