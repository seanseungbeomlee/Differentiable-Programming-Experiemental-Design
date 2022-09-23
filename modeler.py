import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 10,
                      out_channels = 64,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1,
                      padding_mode = 'reflect'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2,
                         padding=0,
                         dilation=1),
            nn.Conv2d(in_channels = 64,
                      out_channels = 64,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1,
                      padding_mode = 'reflect'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=1,
                         padding=1,
                         dilation=1),
            nn.ConvTranspose2d(in_channels = 64,
                               out_channels = 1,
                               kernel_size = 4,
                               stride=2,
                               padding=1,
                               output_padding=0,
                               dilation=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        self.models = []
        self.parameters = []
        for ii  in range(10):
            _model_ = CNNModule()
            self.models.append(_model_)
            self.parameters += list(_model_.parameters())

    def forward(self, x):
        outputs = []
        for ii in range(len(self.models)):
            outputs.append(self.models[ii](x))
        return torch.cat(outputs, 1)
    


mymodel = MyClassifier()
print(mymodel.models[0])
_input = torch.randn((1,10,10,10))
_target = torch.randint(0, 2, _input.shape, dtype=torch.float32)
optimizer = optim.Adam(mymodel.parameters, lr=1.e-3)
loss_func = nn.BCELoss()
mymodel.train()
for epoch in range(5):
    optimizer.zero_grad()
    _output = mymodel.forward(_input)
    _loss = loss_func(_output, _target)
    _loss.backward()
    optimizer.step()
    print(epoch, _loss.item())

print(_input.shape)
print(_target.shape)

print(_output)
print(_target)


# plt.imshow(_output[0][0].detach().numpy())
# plt.show()
# plt.close()
# plt.imshow(_target[0][0].detach().numpy())
# plt.show()