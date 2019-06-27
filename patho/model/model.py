from torch.nn import BCELoss
from torch.optim import Adam
import torch


class Model:
    def __init__(self, net):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net = net
        self.net.to(self.device)
        self.criterion = BCELoss()
        self.optimizer = Adam(self.net.parameters(), lr=1e-2)

    def train(self, trainloader, EPOCH=100):
        for epoch in range(EPOCH):

            running_loss = 0.0
            for ind, (image, segmentation_map) in enumerate(trainloader, 0):
                image, segmentation_map = image.to(self.device), segmentation_map.to(self.device)

                self.optimizer.zero_grad()

                output_map = self.net(image)
                loss = self.criterion(output_map, segmentation_map)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if ind % 10 == 9:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, ind + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished training!')
