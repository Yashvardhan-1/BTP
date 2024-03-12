import torch
import torch.nn as nn

class LeNet(nn.Module):    
    def __init__(self, out_classes=10):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(400,120),  #in_features = 16 x5x5 
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84, out_classes),
            # nn.Softmax(dim=1)
        )
        
    def forward(self,x): 
        a1=self.feature_extractor(x)
        # print(a1.shape)
        a1 = torch.flatten(a1,1)
        a2=self.classifier(a1)
        return a2