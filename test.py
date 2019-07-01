from patho import Model, UNET, DataLoader
import torchvision.transforms as tf
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, ToPILImage, ToTensor

trans = tf.Compose([RandomHorizontalFlip(), RandomVerticalFlip(),
                    ColorJitter()])

data_loader = DataLoader("patho/data/crc", "images",
                         "masks", batch_size=1, transforms=trans).getInstance()

unet = UNET()
model = Model(unet)

model.train(data_loader, EPOCH=1)
