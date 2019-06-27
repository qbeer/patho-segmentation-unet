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
                images, segmentation_maps = data
                
                optimizer.zero_grad()
                
                output_maps = self.net(images)
                loss = self.criterion(output_maps, segmentation_maps)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished training!')
