from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class Model:
    def __init__(self, net):
        self.net = net
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.net.parameters(), lr=1e-4)

    def train(self, trainloader, EPOCH=100):
        for epoch in range(EPOCH):

            running_loss = 0.0
            for ind, data in enumerate(trainloader, 0):
                image, segmentation_map = data

                self.optimizer.zero_grad()

                output_map = self.net(image)
                print('output : ', output_map.shape, segmentation_map.shape)
                loss = self.criterion(output_map, segmentation_map)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished training!')
