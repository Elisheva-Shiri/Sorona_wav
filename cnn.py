#cnn
#dgg -> building a simple NN using convolution layers
from torch.nn import functional as F
from torch import nn
from torchsummary import summary

# This is not the best way, there are better in other videos
class CNNetwork(nn.Module): #the network build by the pytorch fram work so it inheritage from the nn.model mathood (pass data from one layer to the next)
    def __init__(self):
        # Constructor for a vgg simpl,e convolution network
        # architecture: 4 conv blocks / flatten layer / linear layer / softmax (we have 10 diffrent classes)
        super().__init__()
        # Sequential is a container : a bunch of layer and pyton will use it Sequentialy passing teh data from on to another
        # input shape (batch_size, channels, hight, weight)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
            in_channels = 1, # in channle is one
            out_channels = 8, # 8 filters in our convolution layers
            kernel_size = 3, # typical value
            stride = 1, # typical value
            padding = 2 # typical value
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Output size after convolution filter ((w-f+2P)/s)+1 
        # w - weight
        # f - kernel size
        # p - padding
        # s - stride

        self.conv2 = nn.Sequential(
            nn.Conv2d(
            in_channels = 8, # equel to the output of the privieus layer
            out_channels = 16, # double the amount
            kernel_size = 3, 
            stride = 1,
            padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3, 
            stride = 1,
            padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            stride = 1,
            padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.flatten = nn.Flatten()
        # (flatten.size() ,  10) -> (the last conovlutional block output data shape ,  the number of classes we have )
        
        self.linear = nn.Linear(64*5*4, 3) # 3 is teh number of class we have (128 = (64*5*4 )/10 )

        self.softmax = nn.Softmax(dim = 1)
    

    def forward(self, input_data):
        output = self.conv1(input_data)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.flatten(output) #linear layer
        logits = self.linear(output)
        predictions = self.softmax(logits)
        return predictions

        
if __name__ == "__main__":
    #cnn = CNNetwork()
    # A network summery  THE DIFFRENT SHAPE OF THE DATE NETWORK MOVING FROM ONE LAYER TO THE NEXT
    #  summery(the model, shape of the input = input size) -> input_size = (shape of transformetion) 
    # by the urban sound mel- spectogram (number of chnnel, the frequensy = number of mel_bands, the time axis)
    #summary(cnn, (1, 64, 44))
    # summary(cnn.cuda(), (1, 64, 44)) to work on the gpu

    # MaxPool2d-12 last elejment of the last convolutional block

    model = CNNetwork()
    # summary(model, (10, 100))


