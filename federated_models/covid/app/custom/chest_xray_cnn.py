import torch
import torch.nn as nn
import torch.nn.functional as F

class CXRCNN(nn.Module):
    def __init__(self):
        super(CXRCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(350464, 4800),
            nn.ReLU(inplace=True),
            nn.Linear(4800, 514),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(514, 1),
            nn.Softmax(dim=1),
              
        )

    def forward(self, x: float):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
        
class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16384, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        a = self.conv_layer(x)        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x        
    
# Let's try a pre-trained ResNet Model
from torchvision.models import resnet50, ResNet50_Weights

def Mod_ResNet(seed=0):
    """
    """
    # Using pretrained weights, we make a transfer learning
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Frize all parameters, these are already trained
    for param in model.parameters():
        param.requires_grad = False   
    
    # Fix seed manually for training reproducibility
    torch.manual_seed(seed)
    # Let's add the final classification layer with a binary classification and a Sigmoid function  
    model.fc = nn.Sequential(
               nn.Linear(2048, 1),
               nn.Sigmoid()
               )
               
    for fc_par in model.fc.parameters():
        fc_par.requires_grad = True   
        
    #print(list(model.fc.parameters()), len(list(model.fc.parameters())[0][0]))                

    return model        
        
if __name__=='__main__':
    # Import the necessary libraries
    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import convert_image_dtype
    import numpy as np
  
    # Read a PIL image
    image = Image.open('/home/orobix/Projects/bracco/bracco-test/jobs/bracco-test/app/custom/test_image.png')
  
    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
                           transforms.Grayscale(num_output_channels=3),
                           transforms.ToTensor(),
                           transforms.Resize((128, 128)),
                ])
  
    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)
    #img_tensor = convert_image_dtype(img_tensor)
  
    # print the converted Torch tensor
    print(img_tensor)
 
    # Shape
    print(img_tensor.shape)
    
    # NN   
    model = Mod_ResNet()
    output = model(img_tensor.unsqueeze(0))
    print(output.shape)  
    print(model(img_tensor.unsqueeze(0)))    
    
        
