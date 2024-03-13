from lib import *
from image_transform import ImageTransform
from config import *
from utils import make_datapath_list, train_model
from dataset import MyDataset

train_list = make_datapath_list("train")
val_list = make_datapath_list("val")

# dataset
train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase="train")
val_dataset = MyDataset(val_list, transform=ImageTransform(resize, mean, std), phase="val")

# dataloader
batch_size = 4

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

dataloader_dict = { "train": train_dataloader, "val": val_dataloader }

# Network
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

net.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)

# loss
criterior = nn.CrossEntropyLoss()

# optimizer
params_to_update = []

update_params_name = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_params_name:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

# training
train_model(net, dataloader_dict, criterior, optimizer, num_epoch)