from torch import nn
import numpy as np
import torch
import math

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))




# Define model
class LeakyreluNet_2_bm_1(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=256, n_hidden_2=600, n_hidden_3=256, n_hidden_4=128, n_hidden_5=64, out_dim=2):
        super(LeakyreluNet_2_bm_1, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(57, n_hidden_1),
                                     nn.LeakyReLU(),)
   
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU(),)

        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU(),)

        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.LeakyReLU(),)

        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                   nn.LeakyReLU(),)

        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))


    def forward(self, x, noise_scale):
        X1 = x[:, :10]
        X2 = x[:, 10:15]
        X3 = x[:, 15:]

        z2 = (X2-1).long()
        c2 = [2, 7, 2, 2, 3]
        for i in range(5):
            a2 = torch.unsqueeze(z2[:,i],dim=1).to(device)
            b2 = torch.zeros(len(z2[:,i]), c2[i]).to(device).scatter_(1, a2,1)
            if i ==0:
                x2 = b2
            else:
                x2 = torch.cat([x2, b2],1)

        z3 = (X3-1).long()
        c3 = [11, 3, 2, 10, 5]
        for i in range(5):
            a3 = torch.unsqueeze(z3[:,i],dim=1).to(device)
            b3 = torch.zeros(len(z3[:,i]), c3[i]).to(device).scatter_(1, a3, 1)
            if i ==0:
                x3 = b3
            else:
                x3 = torch.cat([x3, b3],1)
        
        x = torch.cat([X1, x2, x3], 1).to(device)


        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)

        noise1 = np.random.normal(0, math.sqrt(noise_scale), layer2_out.size())
        noise1 = torch.from_numpy(noise1).float().cuda()
        layer2_out = (layer2_out + noise1).clone().detach().requires_grad_(True)


        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer6_out



class LeakyreluNet_2_bm_2(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=256, n_hidden_2=600, n_hidden_3=256, n_hidden_4=128, n_hidden_5=64, out_dim=2):
        super(LeakyreluNet_2_bm_2, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(57, n_hidden_1),
                                     nn.LeakyReLU(),)
   
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU(),)

        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU(),)

        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.LeakyReLU(),)

        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                   nn.LeakyReLU(),)

        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))


    def forward(self, x, noise_scale):
        X1 = x[:, :10]
        X2 = x[:, 10:15]
        X3 = x[:, 15:]

        z2 = (X2-1).long()
        c2 = [2, 7, 2, 2, 3]
        for i in range(5):
            a2 = torch.unsqueeze(z2[:,i],dim=1).to(device)
            b2 = torch.zeros(len(z2[:,i]), c2[i]).to(device).scatter_(1, a2,1)
            if i ==0:
                x2 = b2
            else:
                x2 = torch.cat([x2, b2],1)

        z3 = (X3-1).long()
        c3 = [11, 3, 2, 10, 5]
        for i in range(5):
            a3 = torch.unsqueeze(z3[:,i],dim=1).to(device)
            b3 = torch.zeros(len(z3[:,i]), c3[i]).to(device).scatter_(1, a3, 1)
            if i ==0:
                x3 = b3
            else:
                x3 = torch.cat([x3, b3],1)
        
        x = torch.cat([X1, x2, x3], 1).to(device)


        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)

        noise1 = np.random.normal(0, math.sqrt(noise_scale), layer2_out.size())
        noise1 = torch.from_numpy(noise1).float().cuda()

        layer3_out = self.layer3(layer2_out + noise1)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer6_out




# Define model
class LeakyreluNet_2_bm_head_1(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=256, n_hidden_2=200, n_hidden_3=128, n_hidden_4=64, n_hidden_5=32, out_dim=2):
        super(LeakyreluNet_2_bm_head_1, self).__init__()
                                
        self.layer1_1 = nn.Sequential(nn.Linear(10, n_hidden_1),
                                     nn.LeakyReLU(),
                                     nn.Linear(n_hidden_1, n_hidden_2),
                                     nn.LeakyReLU(),)

        self.layer1_2 = nn.Sequential(nn.Linear(16, n_hidden_1),
                                     nn.LeakyReLU(),
                                     nn.Linear(n_hidden_1, n_hidden_2),
                                     nn.LeakyReLU(),)

        self.layer1_3 = nn.Sequential(nn.Linear(31, n_hidden_1),
                                     nn.LeakyReLU(),
                                     nn.Linear(n_hidden_1, n_hidden_2),
                                     nn.LeakyReLU(),)

   
        self.layer2 = nn.Sequential(nn.Linear(3*n_hidden_2, n_hidden_2),
                                    nn.LeakyReLU(),)

        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU(),)

        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.LeakyReLU(),)

        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim),
                                    )



    def forward(self, x, noise_scale):
        X1 = x[:, :10]
        X2 = x[:, 10:15]
        X3 = x[:, 15:]

        z2 = (X2-1).long()
        c2 = [2, 7, 2, 2, 3]
        for i in range(5):
            a2 = torch.unsqueeze(z2[:,i],dim=1).to(device)
            b2 = torch.zeros(len(z2[:,i]), c2[i]).to(device).scatter_(1, a2,1)
            if i ==0:
                x2 = b2
            else:
                x2 = torch.cat([x2, b2],1)

        z3 = (X3-1).long()
        c3 = [11, 3, 2, 10, 5]
        for i in range(5):
            a3 = torch.unsqueeze(z3[:,i],dim=1).to(device)
            b3 = torch.zeros(len(z3[:,i]), c3[i]).to(device).scatter_(1, a3, 1)
            if i ==0:
                x3 = b3
            else:
                x3 = torch.cat([x3, b3],1)
        


        layer1_out_1 = self.layer1_1(X1) 
        layer1_out_2 = self.layer1_2(x2) 
        layer1_out_3 = self.layer1_3(x3) 
        

        noise1 = np.random.normal(0, math.sqrt(noise_scale), layer1_out_1.size())
        noise1 = torch.from_numpy(noise1).float().cuda()
        noise2 = np.random.normal(0, math.sqrt(noise_scale), layer1_out_2.size())
        noise2 = torch.from_numpy(noise2).float().cuda()
        noise3 = np.random.normal(0, math.sqrt(noise_scale), layer1_out_3.size())
        noise3 = torch.from_numpy(noise3).float().cuda()


        layer1_out_1 = (layer1_out_1 + noise1).clone().detach().requires_grad_(True)
        layer1_out_2 = (layer1_out_2 + noise2).clone().detach().requires_grad_(True)
        layer1_out_3 = (layer1_out_3 + noise3).clone().detach().requires_grad_(True)


        x = torch.cat([layer1_out_1, layer1_out_2, layer1_out_3], 1).to(device)


        layer2_out = self.layer2(x)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)

        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer5_out



# Define model
class LeakyreluNet_2_bm_head_2(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=256, n_hidden_2=200, n_hidden_3=128, n_hidden_4=64, n_hidden_5=32, out_dim=2):
        super(LeakyreluNet_2_bm_head_2, self).__init__()
                                
        self.layer1_1 = nn.Sequential(nn.Linear(10, n_hidden_1),
                                     nn.LeakyReLU(),
                                     nn.Linear(n_hidden_1, n_hidden_2),
                                     nn.LeakyReLU(),)

        self.layer1_2 = nn.Sequential(nn.Linear(16, n_hidden_1),
                                     nn.LeakyReLU(),
                                     nn.Linear(n_hidden_1, n_hidden_2),
                                     nn.LeakyReLU(),)

        self.layer1_3 = nn.Sequential(nn.Linear(31, n_hidden_1),
                                     nn.LeakyReLU(),
                                     nn.Linear(n_hidden_1, n_hidden_2),
                                     nn.LeakyReLU(),)

   
        self.layer2 = nn.Sequential(nn.Linear(3*n_hidden_2, n_hidden_2),
                                    nn.LeakyReLU(),)

        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU(),)

        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.LeakyReLU(),)

        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim),
                                    )



    def forward(self, x, noise_scale):
        X1 = x[:, :10]
        X2 = x[:, 10:15]
        X3 = x[:, 15:]

        z2 = (X2-1).long()
        c2 = [2, 7, 2, 2, 3]
        for i in range(5):
            a2 = torch.unsqueeze(z2[:,i],dim=1).to(device)
            b2 = torch.zeros(len(z2[:,i]), c2[i]).to(device).scatter_(1, a2,1)
            if i ==0:
                x2 = b2
            else:
                x2 = torch.cat([x2, b2],1)

        z3 = (X3-1).long()
        c3 = [11, 3, 2, 10, 5]
        for i in range(5):
            a3 = torch.unsqueeze(z3[:,i],dim=1).to(device)
            b3 = torch.zeros(len(z3[:,i]), c3[i]).to(device).scatter_(1, a3, 1)
            if i ==0:
                x3 = b3
            else:
                x3 = torch.cat([x3, b3],1)
        


        layer1_out_1 = self.layer1_1(X1) 
        layer1_out_2 = self.layer1_2(x2) 
        layer1_out_3 = self.layer1_3(x3) 
        

        noise1 = np.random.normal(0, math.sqrt(noise_scale), layer1_out_1.size())
        noise1 = torch.from_numpy(noise1).float().cuda().requires_grad_(False)
        noise2 = np.random.normal(0, math.sqrt(noise_scale), layer1_out_2.size())
        noise2 = torch.from_numpy(noise2).float().cuda().requires_grad_(False)
        noise3 = np.random.normal(0, math.sqrt(noise_scale), layer1_out_3.size())
        noise3 = torch.from_numpy(noise3).float().cuda().requires_grad_(False)

        x = torch.cat([(layer1_out_1+noise1), (layer1_out_2+noise2), (layer1_out_3+noise3)], 1).to(device)


        layer2_out = self.layer2(x)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)

        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer5_out

