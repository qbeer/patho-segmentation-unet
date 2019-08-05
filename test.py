from patho import Model, UNET_VGG, DataLoader
import matplotlib.pyplot as plt
import numpy as np

data_loader = DataLoader("patho/data/crc",
                         "images",
                         "masks",
                         batch_size=3,
                         output_size=572).getInstance()

unet = UNET_VGG()
unet.make_parallel()

model = Model(unet, lr=5e-6, with_jaccard=True, load_model=True)

for ind_, (imgs, masks) in enumerate(data_loader):
    predicted_masks = model.net(imgs.cuda()).cpu().detach().numpy()
    imgs, masks = imgs.cpu().detach().numpy(), masks.cpu().detach().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

    for ind, ax in enumerate(axes.flatten(), 0):
        if ind % 3 == 0:
            ax.imshow(imgs[ind // 3].transpose((1, 2, 0)))
        elif ind % 3 == 1:
            ax.imshow(masks[ind // 3].transpose((1, 2, 0)).reshape(572, 572))
        else:
            ax.imshow(predicted_masks[ind // 3].transpose(
                (1, 2, 0)).reshape(572, 572))

    fig.tight_layout()
    plt.savefig("raw_" + str(ind_) + ".png")

    predicted_masks[predicted_masks >= 0.75] = 1.
    predicted_masks[predicted_masks < 0.75] = 0.

    fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

    for ind, ax in enumerate(axes.flatten(), 0):
        if ind % 3 == 0:
            ax.imshow(imgs[ind // 3].transpose((1, 2, 0)))
        elif ind % 3 == 1:
            ax.imshow(masks[ind // 3].transpose((1, 2, 0)).reshape(572, 572))
        else:
            ax.imshow(predicted_masks[ind // 3].transpose(
                (1, 2, 0)).reshape(572, 572))

    fig.tight_layout()
    plt.savefig("075cut" + str(ind_) + ".png")
