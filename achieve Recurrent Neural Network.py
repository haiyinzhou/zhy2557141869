import torch

input_size = 4
hidden_size = 4
num_layers = 1
batch_size = 1
seq_len = 5

x = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]
y = [3,1,2,3,2]

x = torch.Tensor(x).view(seq_len,batch_size,input_size)
y = torch.LongTensor(y)

class RNN(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers=1):
        super(RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size=self.input_size,hidden_size=hidden_size,num_layers=num_layers)

    def forward(self,x):
        hidden = torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        out,_ = self.rnn(x,hidden)
        out = out.view(-1,self.hidden_size)
        return out

net = RNN(input_size=input_size,hidden_size=hidden_size,batch_size=batch_size,num_layers=num_layers)
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(),lr=0.001)

for epoch in range(10):
    output = net(x)
    loss = loss_func(output,y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print("{}epoch's loss is:{}".format(epoch+1,loss))