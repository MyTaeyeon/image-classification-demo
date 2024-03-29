from lib import *

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

num_epoch = 3

save_path = "./weight_transfer_learning.pth"