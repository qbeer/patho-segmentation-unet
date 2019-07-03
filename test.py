from patho import Model, UNET, DataLoader

data_loader = DataLoader("patho/data/crc", "images",
                         "masks", batch_size=1).getInstance()

unet = UNET()
model = Model(unet)

print("Init done.")

model.train(data_loader, EPOCH=1)
