from ..custom_loss import BCELossWithJaccard
from torch.nn import BCELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from ..model import UNET
import torch
import os


class Model:
    def __init__(self, net, lr=5e-3, with_jaccard=False, load_model=False):
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net = net
        if load_model:
            self.net = UNET()
            try:
                self.net.load_state_dict(torch.load("patho/data/model.pt"))
            except RuntimeError:
                self.net.make_parallel()
                self.net.load_state_dict(torch.load("patho/data/model.pt"))
            finally:
                self.net.eval()
        self.net.to(self.device)
        self.criterion = BCELoss()
        if with_jaccard:
            self.criterion = BCELossWithJaccard()
        self.optimizer = SGD(self.net.parameters(), lr=lr, momentum=0.99)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.1)

    def train(self, data_loader, EPOCH=10):
        for epoch in range(EPOCH):

            loss_on_epoch_end = 0.0
            running_loss = 0.0
            for ind, (image, segmentation_map) in enumerate(data_loader, 0):
                image, segmentation_map = image.to(
                    self.device), segmentation_map.to(self.device)

                self.scheduler.zero_grad()

                output_map = self.net(image)
                loss = self.criterion(
                    output_map.view(-1), segmentation_map.view(-1))
                loss.backward()
                self.scheduler.step()

                running_loss += loss.item()
                loss_on_epoch_end += loss.item()
                if ind % 10 == 9:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, ind + 1, running_loss / 10))
                    running_loss = 0.0
            loss_on_epoch_end /= (ind + 1)  # batch average loss
            print("Average 10-batch loss on epoch end : %.5f" % loss_on_epoch_end)

        torch.save(self.net.state_dict(), "patho/data/model.pt")
        print('Finished training!')
