import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, dropout=0.5):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # embedding layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM layer with droputout
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, dropout=dropout, batch_first=True)
        # Linear layer that maps the hidden state output dim to the # of words as output, vocab_size
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, features, captions):
        
        # add one more dimension for features as LSTM input
        features = features.view(-1, 1, self.embed_size)
        
        captions = self.embed(captions)
        
        # Discard the <end> token and Stack the features and captions as inputs
        inputs = torch.cat((features, captions[:, :-1,:]), dim=1)
        
        #lstm_output shape : (batch_size, caption length, hidden_size)
        lstm_output, lstm_hidden = self.lstm(inputs)
        
        # Fully connected layer to turn the output into vectors in the size  (batch_size, caption length, vocab_size)
        output = self.fc(lstm_output)
        
        output_scores = self.softmax(output)
        return output_scores
   

    def sample(self, inputs, states=None, max_len=20):
        sampled_ids = [] 
        for i in range(max_len): 
            hiddens, states = self.lstm(inputs, states) 
            outputs = self.fc(hiddens.squeeze(1)) 
            predicted = outputs.max(1)[1] 
            sampled_ids.append(predicted.data[0].item()) 
            inputs = self.embed(predicted) 
            inputs = inputs.unsqueeze(1) 
        return sampled_ids
