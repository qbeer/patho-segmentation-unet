from patho import Model, UNET, DataLoader

data_loader = DataLoader("patho/data", "resized_images",
                         "resized_masks").getInstance()

unet = UNET()
model = Model(unet)
