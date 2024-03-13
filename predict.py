from config import *
from lib import *
from utils import load_model
import image_transform

class_index = ["ants", "bees"]

class Predictor():
    def __init__(self, class_index):
        self.clas_index = class_index

    def predict_max(self, output): # [0.9, 0.1]
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.clas_index[max_id]
        return predicted_label


predictor = Predictor(class_index)

def predict(img):
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.eval()

    model = load_model(net, save_path)

    transform = image_transform.ImageTransform(resize, mean, std)
    img = transform(img, phase="test")
    img = img.unsqueeze_(0)

    output = model(img)
    label = predictor.predict_max(output)

    return label

if __name__ == "__main__":
    path_img = "./data/mini-beebee.jpg"
    img = Image.open(path_img)

    label = predict(img)

    plt.imshow(img)
    plt.title(f"predicted: {label}")
    plt.show()
