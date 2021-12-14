import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import numpy as np

class RandomModel(nn.Module):

    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self._num_classes = num_classes

    def forward(self, x, lengths):
        y = torch.ones(x.shape[0], self._num_classes)
        return y/self._num_classes

    def sample(self, out):
        return torch.randint(self._num_classes, (out.shape[0], ))
    
class LSTMModel(nn.Module):
    
    def __init__(self, num_classes, input_size, lstm_dim=100, n_layers=1, hidden_dim=128):

        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_dim, num_layers=n_layers, batch_first=True)

        self.output_layer = nn.Sequential(
            nn.Linear(lstm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x, lengths):
        '''
        Inputs:
            x : Tensor of shape batch_size x time x input_size
            lengths : Tensor of shape batch_size x 1 

        Outputs:
            class_probs : Tensor of batch_size x num_classes
        '''

        # Propagate input through LSTM

        lstm_output = self.lstm(x)[0][torch.arange(x.shape[0]), lengths-1] # batch_size x lstm_dim
   
        return self.output_layer(lstm_output)
    
    def sample(self, out):
        return np.argmax(out.clone().detach(), axis=-1)

class SignalResize(nn.Module):

    def __init__(self, output_size):
        self._output_size = output_size


    def forward(self, x, lengths):

        '''
        Input is x is batch x time x features already padded
        lentghts is batch x 1, input sizes of each signal (excluding padding)
        '''

        # Compute what the filter sizes should be for each input so that the output size is constant
        filter_sizes = lengths - self.output_size + 1

        # Each entry of repeats alternatively represents how many ones you need and then zeros, for each input in i in the batch we need filter_sizes[i] ones and then filter_sizes.max() - filter_sizes[i] zeros to create the padded filter
        repeats = torch.repeat_interleave(filter_sizes, 2)
        repeats[1::2] *= -1
        repeats[1::2] += filter_sizes.max()

        # Create averaging filters of needed length (with padding) and transform in matrix of shape batch x max_filter_size
        filters = torch.repeat_interleave(
            torch.Tensor([1., 0.]*x.shape[0]),
            repeats
        ).view(x.shape[0], -1) / filter_sizes[:, None]

        # Convolve
        return torch.squeeze(
            F.conv2d(
                x[None],
                filters.view(-1, 1, filters.shape[1], 1),
                groups = x.shape[0]
            )
        )

class CNN_LSTMModel(nn.Module):

    def __init__(self, num_classes, input_size, kernel_size=3,num_channels=2, stride=1, padding=0, lstm_dim=100, n_layers=1, hidden_dim=128):

        super().__init__()
        
        self.kernel_size=kernel_size

        self.cnn_layer = nn.Conv2d(1,num_channels, kernel_size=(kernel_size,1), stride=stride, padding=padding)
        
        self.lstm_layer = LSTMModel(num_classes, input_size*num_channels, lstm_dim, n_layers, hidden_dim)
        
    def forward(self, x, lengths):
        '''
        Inputs:
            x : Tensor of shape batch_size x time x input_size
            lengths : Tensor of shape batch_size x 1 

        Outputs:
            class_probs : Tensor of batch_size x num_classes
        '''

        # Propagate input through CNN layer first


        cnn_out=self.cnn_layer(x[:, None])

        # Propagate CNN output to LSTM layer second
        cnn_out = cnn_out.permute(0,2,1,3).reshape(cnn_out.shape[0], cnn_out.shape[2],-1)
   
        return self.lstm_layer(cnn_out, (lengths-1)-self.kernel_size+1)
    
    def sample(self, out):
        return np.argmax(out.clone().detach(), axis=-1)

    
    
class dense_CNNModel(nn.Module):

    def __init__(self, output_size, num_classes, out_features , input_size, **kwargs):

        super(dense_CNNModel,self).__init__()
        self._output_size = output_size

        self.bn=nn.BatchNorm1d(input_size)
        self.relu = nn.ReLU(inplace = True)
        self.conv1=nn.Conv1d(in_channels=input_size, out_channels=out_features , kernel_size=1, stride=1, padding=0)
        self.conv2=nn.Conv1d(in_channels=out_features, out_channels=out_features , kernel_size=1, stride=1, padding=0)
        self.conv3=nn.Conv1d(in_channels=2*out_features, out_channels=out_features , kernel_size=1, stride=1, padding=0)
        self.conv4=nn.Conv1d(in_channels=3*out_features, out_channels=out_features , kernel_size=1, stride=1, padding=0)
        self.conv5=nn.Conv1d(in_channels=4*out_features, out_channels=out_features , kernel_size=1, stride=1, padding=0)
        self.finalconv=nn.Conv1d(in_channels=out_features, out_channels=out_features , kernel_size=1, stride=1, padding=0)
        self.drop=nn.Dropout(0.5)
        self.pool=nn.MaxPool1d(kernel_size=1, stride=1)


        #linear classifcation layer
        self.linear_layers = nn.Sequential(
                #convolutional arithmetic is needed for an exact input expression
                    nn.Linear(4200 , num_classes),
                #normalize classification probabilities
                    nn.Softmax(dim=1)
        )



    def forward(self, y , lengths):

        #Create averaging filters for each batch
        #to standardize input timeseries lengths
        filter_sizes = lengths - self._output_size + 1

        repeats = torch.repeat_interleave(filter_sizes, 2)
        repeats[1::2] *= -1
        repeats[1::2] += filter_sizes.max()

        filters = torch.repeat_interleave(
            torch.Tensor([1.,0.]*y.shape[0]),
            repeats
        ).view(y.shape[0], -1)/filter_sizes[:,None]

        #convolve averaging filters onto time-series data
        y = torch.squeeze(
            F.conv2d(
                y[None],
                filters.view(-1,1,filters.shape[1],1),
                groups=y.shape[0]
            )
        )

        #reordering format for CNN Layers
        y = y.permute(0,2,1)
        #pass through CNN layers

        #Dense CNN layers w/ optional dropout and final convolutional layer
        y = self.bn(y)
        conv1 = self.relu(self.conv1(y))
        conv2 = self.relu(self.conv2(conv1))
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        #c2_dense = self.drop(c2_dense)
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        #c3_dense = self.drop(c3_dense)
        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
        #c4_dense = self.drop(c4_dense)
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
        #c5_dense = self.drop(c5_dense)
        #c5_dense=self.relu(self.finalconv(c5_dense))

        y = self.pool(c5_dense)

        #Linearizing data for classification
        y = y.view(y.shape[0], -1)
        #classify each example
        y=self.linear_layers(y)
        return y

    def sample(self, out):
        #return most probable classification
        return np.argmax(out.clone().detach(),axis=1)
