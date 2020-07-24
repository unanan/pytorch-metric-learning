import torch.nn as nn
import torch.nn.functional as F

class EmbClsNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(EmbClsNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        embeddings = self.embedding_net(x)
        embeddings = self.nonlinear(embeddings)
        scores = F.log_softmax(self.fc1(embeddings), dim=-1)
        return embeddings, scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))