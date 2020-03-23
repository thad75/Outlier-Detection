
import torch
import torch.nn as nn
import torch.nn.parallel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




class Discriminator(nn.Module):
  def __init__(self, ngpu, latent_size):

    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.latent_size = latent_size
#for x
    self.input_x= nn.Sequential (
        nn.Conv2d(in_channels= 1,
                  out_channels= 64,
                  kernel_size= 4, #28-5+1=24
                  stride = 2,
                  padding = (4-1)//2),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels= 64,
                  out_channels= 64,
                  kernel_size= 4, #28-3+1=20
                  stride = 2,padding = (4-1)//2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1),
        )
#for z (d(Z))
    self.input_z= nn.Sequential(
        nn.Linear(latent_size,512, bias = True),
        nn.LeakyReLU(0.1),
        )

#then we compute d(x,z) by concateantion and we also need the intermediate layer
#for loss Calculation

    #here we compute the intermediate layer value
    self.concaxz = nn.Sequential(
        nn.Linear(3648,1024, bias = True),
        nn.LeakyReLU(0.1),
        # nn.Dropout(0.5),
    )
     
    self.output_discri = nn.Sequential(
    nn.Linear(in_features = 1024,out_features = 1, bias = True),
    nn.Sigmoid(),
    )
    


  def forward(self,z,x):
    
    x=x.view(-1,1,28,28)
    output_x= self.input_x(x)
    output_x = output_x.view(-1,7*7*64)
    
    output_z = self.input_z(z)
    
    output_intermediate_features = self.concaxz(torch.cat([output_x,output_z], dim =1))
    output = self.output_discri(output_intermediate_features)
    return output.squeeze(), output_intermediate_features.view(x.size()[0],-1)

class Generator(nn.Module):
  def __init__(self, ngpu, latent_size):

    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.latent_size = latent_size
    self.dense = nn.Sequential (
        
         nn.Linear(latent_size,1024),
         nn.ReLU(True),
        
##définir les inputs genre y ensuite définir le discrimaintaor puis les scores
        nn.Linear(1024,7*7*128),
        #nn.BatchNorm2d(7*7*128),
        nn.ReLU(True)
        )
    
    self.convlayer= nn.Sequential(
         nn.ConvTranspose2d(128,
                           out_channels= 64,
                           kernel_size = 4,
                           stride = 2,padding = (4-1)//2, bias = True),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.ConvTranspose2d(64,
                           out_channels= 1,
                           kernel_size = 4,
                           stride = 2,padding = (4-1)//2
                           , bias = True),
        nn.Tanh()
    )

  def forward(self, input):
    #input = input.view(-1, latent_size)
    # print("Generatpr "+ str(input.shape))
    dense = self.dense(input)
    #print('dense ' +str(dense.shape))
    dense = dense.view(-1,128,7,7)
    output = self.convlayer(dense)
    # print('generator' +str(dense.shape))
    return output



class Encoder(nn.Module):
  def __init__(self,ngpu,latent_size):
    super(Encoder,self).__init__()
    self.latent_size=latent_size
    self.npgu = ngpu
    self.layer_1= nn.Sequential(

#layer 1    
        nn.Conv2d(in_channels= 1, #image is in black and white
                  out_channels= 32,
                  kernel_size= 3,
                  stride = 1,
                  padding = (3-1)//2 ),
        # nn.BatchNorm2d(32),
        nn.LeakyReLU(0.1),

#layer 2
        nn.Conv2d(in_channels= 32,
                  out_channels =  64,
                  kernel_size = 3,
                  stride= 2,
                  padding = (3-1)//2, bias = True),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1),

#layer 3
        nn.Conv2d(64,
                  128,
                  3, 
                  stride = 2,
                  padding = (3-1)//2,bias= True),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.1),)



    self.layer_2 = nn.Linear(6272,latent_size, bias = True)
        # nn.Conv2d(128,latent_size,3),
        # nn.BatchNorm2d(latent_size),
        # nn.LeakyReLU(0.01)
    


  def forward(self, input):
    # print(" ###### ")
    # print("Encoder :")
    # print(input.shape)
    input = input.view(-1,1,28,28)
    output = self.layer_1(input)
    output= output.view(output.size(0),-1)
    output1 = self.layer_2(output)
    # print(output1.shape)
    # print(" ###### ")
    
    return output1   #,output3.view(batch_size, -1), output2.view(batch_size, -1), output1.view(batch_size, -1)


 