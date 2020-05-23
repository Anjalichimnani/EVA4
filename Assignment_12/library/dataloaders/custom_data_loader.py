from imports.imports_eva import *

class custom_data_loader:
    def __init__():
        super().__init__()
    
    def get_def_data_transform():
        data_transform = transforms.Compose([
                                                    transforms.RandomRotation(5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.4770698,  0.44741154, 0.3993873 ],
                                                    std=[0.27973273, 0.27094352, 0.28073433])
        ])
        return data_transform

    def custom_data_set_image_folder (root_path, folder, transform=None):
        return datasets.ImageFolder(root=root_path + folder,
                                           transform=transform)
                                           
    def data_loader (dataset, batch_size, num_workers, shuffle=False):
        return torch.utils.data.DataLoader (dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)