from patho import Model, UNET, DataLoader

data_loader = DataLoader("patho/data", "resized_images",
                         "resized_masks", batch_size=1).getInstance()

unet = UNET()
model = Model(unet)

model.train(data_loader, EPOCH=1)
