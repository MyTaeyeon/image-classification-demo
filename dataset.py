import torch.utils.data as data
from PIL import Image

class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == 'train':
            label = img_path[30:34] 
        elif self.phase == 'val':
            label = img_path[28:32]

        label = 0 if label=="ants" else 1
        return img_transformed, label