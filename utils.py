from lib import *
from config import save_path
import time

def make_datapath_list(phase='train'):
    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+"/**/*.jpg")

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list

def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
                
            epoch_loss = 0.0
            epoch_corrects = 0
            
            if (epoch == 0) and (phase == "train"):
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]):
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                epoch_loss += loss.item()*inputs.size(0)
                epoch_corrects += torch.sum(preds==labels.data)
            
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}")

    torch.save(net.state_dict(), save_path)

def load_model(net, model_path):
    load_weights = torch.load(model_path,  map_location={"cuda:0": "cpu"})
    net.load_state_dict(load_weights)
    return net
