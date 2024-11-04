import torch
import torch.nn as nn
import torch.nn.functional as F


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(residual_block, self).__init__()
        self.layer1=self.Conv(in_channels,40,dilation_rate)
        self.layer2=self.Conv(40,out_channels,dilation_rate)
        
    def Conv(self, in_channels, out_channels, dilation_rate): 
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate), #padding = dilation rate for maintaining constant dimensions
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
         )
    
    def forward(self,x):
        result1=self.layer1(x)
        result2=self.layer2(result1)
        return result1+result2    
    
class RDNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RDNet, self).__init__()
#       left side   
        self.standard_block1=self.standard_block(in_channels,40)
        self.residual_block2=residual_block(40,40,2)
        self.residual_block3=residual_block(40,40,4)
        self.residual_block4=residual_block(40,40,8)
        self.residual_block5=residual_block(40,40,16)
               
#       bridge
        self.dropout1=nn.Dropout(p=0.7)
        self.residual_block6=residual_block(40,40,16)
        self.dropout2=nn.Dropout(p=0.7)
        
#       Right side

        self.residual_block7=residual_block(40,40,16)
        self.residual_block8=residual_block(40,40,8)
        self.residual_block9=residual_block(40,40,4)
        self.residual_block10=residual_block(40,40,2)
        self.residual_block11=residual_block(40,40,1)
        self.activation = nn.Softmax(dim=1)  # For multi-class classification

        
#       final layer
        
        self.final_conv = nn.Conv3d(40, out_channels, kernel_size=1)
        
                
    def standard_block(self, in_channels, out_channels, dilation_rate=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
    
    def forward(self,x):
        result1=self.standard_block1(x)
        result2=self.residual_block2(result1)
        result2=result1+result2
        
        result3=self.residual_block3(result2)
        result4=self.residual_block4(result3)
        result5=result3+result4
        
        result6=self.residual_block5(result5)
        result8=self.dropout1(result6)
        result9=self.residual_block6(result8)
        result10=self.dropout2(result9)
        result12=self.residual_block7(result10)
        result14=self.residual_block8(result12)
        result14=result14+result4
        result14=result14+result12
        result15=self.residual_block9(result14)
        result15=result14+result3
        result17=self.residual_block10(result15)
        result17=result17+result15
        result17=result17+result2
        result18=self.residual_block11(result17)
        result18=result18+result1
        
        result=self.final_conv(result18)
        
        return self.activation(result)  

        

