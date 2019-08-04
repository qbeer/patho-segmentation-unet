from patho import Model, UNET_VGG, DataLoader

data_loader = DataLoader("patho/data/crc",
                         "images",
                         "masks",
                         batch_size=2,
                         output_size=572).getInstance()
unet = UNET_VGG()

for img, mask in data_loader:
    break

print(unet.forward(img).shape)
#model = Model(unet)


#print("Init done.")

#model.train(data_loader, EPOCH=1)
