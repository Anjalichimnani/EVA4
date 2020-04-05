from imports.imports_eva import *

class albumentation:
    def __init__(self):
        self.albumentation_transform = Compose([RandomCrop (32, 32),
                                                Cutout(num_holes=3, max_h_size=8, max_w_size=8),
                                                Normalize(
                                                    mean=[0.5, 0.5, 0.5],
                                                    std=[0.5, 0.5, 0.5],
                                                ),
                                                ToTensor()
                                                ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transform(image=img)
        return img['image']


class albumentation_test:
    def __init__(self):
        self.albumentation_transform = Compose([
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensor()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transform(image=img)
        return img['image']


def train_test_loaders(b_size, n_workers=2):
    # Download Train/test data----------------------------------------------------------------
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=albumentation())

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=albumentation_test())

    print('Train Images count', len(trainset))
    print('Test Images count', len(testset))

    # Define data loaders---------------------------------------------------------------------
    # Train data loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size,
                                              shuffle=True, num_workers=n_workers)
    # Test data loader
    testloader = torch.utils.data.DataLoader(testset, batch_size=b_size,
                                             shuffle=False, num_workers=n_workers)

    return trainloader, testloader