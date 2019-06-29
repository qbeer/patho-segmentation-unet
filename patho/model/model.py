from ..custom_loss import BCELossWithJaccard
from torch.nn import BCELoss
from torch.optim import SGD
import torch
import os


class Model:
    def __init__(self, net, lr=1e-3, with_jaccard=False, load_model=False):
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net = net
        if load_model:
            self.net = self.net.load_state_dict(
                torch.load(os.join("patho/data", "model.pth")))
        self.net.to(self.device)
        self.criterion = BCELoss()
        if with_jaccard:
            self.criterion = BCELossWithJaccard()
        self.optimizer = SGD(self.net.parameters(), lr=lr, momentum=0.9)

    def train(self, data_loader, EPOCH=100):
        for epoch in range(EPOCH):

            loss_on_epoch_end = 0.0
            running_loss = 0.0
            for ind, (image, segmentation_map) in enumerate(data_loader, 0):
                image, segmentation_map = image.to(
                    self.device), segmentation_map.to(self.device)

                self.optimizer.zero_grad()

                output_map = self.net(image)
                loss = self.criterion(
                    output_map.view(-1), segmentation_map.view(-1))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                loss_on_epoch_end += loss.item()
                if ind % 10 == 9:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, ind + 1, running_loss / 10))
                    running_loss = 0.0
            loss_on_epoch_end /= (ind + 1)  # batch average loss
            print("Average batch loss on epoch end : %.5f" % loss_on_epoch_end)

        torch.save(self.net.state_dict(), os.join("patho/data", "model.pth"))
        print('Finished training!')
