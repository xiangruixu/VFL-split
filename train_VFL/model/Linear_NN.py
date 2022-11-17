from torch import nn
import numpy as np
import torch

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))




class Client_leakyrelu_4(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, n_hidden_3 = 64):
        super(Client_leakyrelu_4, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,  n_hidden_2),
                                   nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,  n_hidden_2),
                                    )
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_2,  n_hidden_2),
                                    )

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer4_out
      


class Server_LeakyreluNet_4(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, n_hidden_5= 32, out_dim=2):
        super(Server_LeakyreluNet_4, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.LeakyReLU())
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim),
                                   )


    def forward(self, x1, x2, x3):
        x= torch.cat([x1, x2, x3], dim=1)
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()

        return layer6_out

class Server_LeakyreluNet_1(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, n_hidden_5= 32, out_dim=2):
        super(Server_LeakyreluNet_1, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        # self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))
        # self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
        #                             nn.LeakyReLU())
        # self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim),
        #                            )


    def forward(self, x1, x2, x3):
        x= torch.cat([x1, x2, x3], dim=1)
        layer3_out = self.layer3(x)
        # layer4_out = self.layer4(layer3_out)
        # layer5_out = self.layer5(layer4_out)
        # layer6_out = self.layer6(layer5_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()

        return layer3_out
        


class Client_LinearNet_6_leakyrelu_decoder(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64):
        super(Client_LinearNet_6_leakyrelu_decoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1*5),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1*5, n_hidden_1*4),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_1*4, n_hidden_1*3),
                                    nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_1*3, n_hidden_1*2),
                                    nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_1*2, n_hidden_1),
                                    nn.LeakyReLU())
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer6_out


class Client_LinearNet_2_em_cancer(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LinearNet_2_em_cancer, self).__init__()
       
        self.layer1 = nn.Sequential(nn.Linear(in_dim*10, n_hidden_1),
                                   )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                   )


    def forward(self, x, client):
        z = (x-1).long()
        c = [10,10,10]
        for i in range(3):
            # print('z[:,i]', z[:,i])
            a = torch.unsqueeze(z[:,i],dim=1).to(device)
            # print('a', a)
            # print('a.shape', a.shape)
            b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
            # print('b', b)
            if i ==0:
                x = b
            else:
                x = torch.cat([x, b],1)
                # print('x', x)
                
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out



class Client_Leakyrelu_2_em_cancer(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_Leakyrelu_2_em_cancer, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim*10, n_hidden_1),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())


    def forward(self, x, client):
        z = (x-1).long()
        c = [10,10,10]
        for i in range(3):
            # print('z[:,i]', z[:,i])
            a = torch.unsqueeze(z[:,i],dim=1).to(device)
            # print('a', a)
            # print('a.shape', a.shape)
            b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
            # print('b', b)
            if i ==0:
                x = b
            else:
                x = torch.cat([x, b],1)
                # print('x', x)
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out



class Client_Leakyrelu_3_em_cancer(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_Leakyrelu_3_em_cancer, self).__init__()

    
        self.layer1 = nn.Sequential(nn.Linear(in_dim*10, n_hidden_1),
                                    nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_2),
                                    nn.LeakyReLU())


    def forward(self, x, client):
        z = (x-1).long()
        c = [10,10,10]
        for i in range(3):
            a = torch.unsqueeze(z[:,i],dim=1).to(device)
            b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
            if i ==0:
                x = b
            else:
                x = torch.cat([x, b],1)
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out




class Client_LinearNet_2_em_credit(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LinearNet_2_em_credit, self).__init__()


        # self.embedding_layer = nn.Embedding.from_pretrained(a)
        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(7, n_hidden_1),
                                    )

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(7, n_hidden_1),
                                    )

        if client ==3:
            self.layer1 = nn.Sequential(nn.Linear(75, n_hidden_1),
                                   )

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    )


    def forward(self, x, client):
        if client ==3:
            z = (x-1).long()
            c = [2, 4, 3, 11, 11, 11, 11, 11, 11]
            for i in range(9):
                # print('z[:,i]', z[:,i])
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                # print('a', a)
                # print('a.shape', a.shape)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                # print('b', b)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)
                    # print('x', x)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out



class Client_LeakyreluNet_2_em_credit(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LeakyreluNet_2_em_credit, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(7, n_hidden_1),
                                    nn.LeakyReLU())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(7, n_hidden_1),
                                    nn.LeakyReLU())

        if client ==3:
            self.layer1 = nn.Sequential(nn.Linear(75, n_hidden_1),
                                    nn.LeakyReLU())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())


    def forward(self, x, client):
        if client ==3:
            z = (x-1).long()
            c = [2, 4, 3, 11, 11, 11, 11, 11, 11]
            for i in range(9):
                # print('z[:,i]', z[:,i])
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                # print('a', a)
                # print('a.shape', a.shape)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                # print('b', b)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)
                    # print('x', x)
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out



class Client_LeakyreluNet_3_em_credit(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LeakyreluNet_3_em_credit, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(7, n_hidden_1),
                                    nn.LeakyReLU())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(7, n_hidden_1),
                                    nn.LeakyReLU())

        if client ==3:
            self.layer1 = nn.Sequential(nn.Linear(75, n_hidden_1),
                                    nn.LeakyReLU())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_2),
                                    nn.LeakyReLU())


    def forward(self, x, client):
        if client ==3:
            z = (x-1).long()
            c = [2, 4, 3, 11, 11, 11, 11, 11, 11]
            for i in range(9):
                # print('z[:,i]', z[:,i])
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                # print('a', a)
                # print('a.shape', a.shape)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                # print('b', b)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)
                    # print('x', x)
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out






class Client_LinearNet_2_em_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LinearNet_2_em_bm, self).__init__()


        # self.embedding_layer = nn.Embedding.from_pretrained(a)
        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(10, n_hidden_1),
                                    )

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(16, n_hidden_1),
                                    )

        if client ==3:

            self.layer1 = nn.Sequential(nn.Linear(31, n_hidden_1),
                                   )

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    )


    def forward(self, x, client):
        if client ==2:
            z = (x-1).long()
            c = [2, 7, 2, 2, 3]
            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)
        if client ==3:
            z = (x-1).long()
            c = [11, 3, 2, 10, 5]
            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out



class Client_LeakyreluNet_2_em_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LeakyreluNet_2_em_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(10, n_hidden_1),
                                    nn.LeakyReLU())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(16, n_hidden_1),
                                    nn.LeakyReLU())

        if client ==3:
            self.layer1 = nn.Sequential(nn.Linear(31, n_hidden_1),
                                    nn.LeakyReLU())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())


    def forward(self, x, client):
        if client ==2:
            z = (x-1).long()
            c = [2, 7, 2, 2, 3]
            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)
        if client ==3:
            z = (x-1).long()
            c = [11, 3, 2, 10, 5]
            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out



class Client_LeakyreluNet_3_em_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LeakyreluNet_3_em_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(10, n_hidden_1),
                                    nn.LeakyReLU())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(16, n_hidden_1),
                                    nn.LeakyReLU())

        if client ==3:
            self.layer1 = nn.Sequential(nn.Linear(31, n_hidden_1),
                                    nn.LeakyReLU())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_2),
                                    nn.LeakyReLU())


    def forward(self, x, client):
        if client ==2:
            z = (x-1).long()
            c = [2, 7, 2, 2, 3]
            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)
        if client ==3:
            z = (x-1).long()
            c = [11, 3, 2, 10, 5]
            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out



class Client_Leakyrelu_2_em_census(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_Leakyrelu_2_em_census, self).__init__()

        if client == 1:
            self.layer1 = nn.Sequential(nn.Linear(7, n_hidden_1),
                                        nn.LeakyReLU())
        if client == 2:
            self.layer1 = nn.Sequential(nn.Linear(260, n_hidden_1),
                                        nn.LeakyReLU())
        if client == 3:
            self.layer1 = nn.Sequential(nn.Linear(238, n_hidden_1),
                                        nn.LeakyReLU())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())

    def forward(self, x, client):
        if client == 2:
            z = (x).long()
            c = [9, 3, 2, 24, 15, 5, 10, 2, 3, 6, 8, 6,52,47,17,51]
            for i in range(16):
                a = torch.unsqueeze(z[:, i], dim=1).to(device)
                b = torch.zeros(len(z[:, i]), c[i]).to(device).scatter_(1, a, 1)
                if i == 0:
                    x = b
                else:
                    x = torch.cat([x, b], 1)
        if client == 3:
            z = (x).long()
            c = [6, 8, 10, 9, 10, 3, 4, 5, 5, 3, 3, 3, 2,38,43,43,43]
            for i in range(17):
                a = torch.unsqueeze(z[:, i], dim=1).to(device)
                b = torch.zeros(len(z[:, i]), c[i]).to(device).scatter_(1, a, 1)
                if i == 0:
                    x = b
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out


class Client_Leakyrelu_3_em_census(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_Leakyrelu_3_em_census, self).__init__()

        if client == 1:
            self.layer1 = nn.Sequential(nn.Linear(7, n_hidden_1),
                                        nn.LeakyReLU())
        if client == 2:
            self.layer1 = nn.Sequential(nn.Linear(260, n_hidden_1),
                                        nn.LeakyReLU())
        if client == 3:
            self.layer1 = nn.Sequential(nn.Linear(238, n_hidden_1),
                                        nn.LeakyReLU())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_2),
                                    nn.LeakyReLU())

    def forward(self, x, client):
        if client == 2:
            z = (x).long()
            c = [9, 3, 2, 24, 15, 5, 10, 2, 3, 6, 8, 6,52,47,17,51]
            for i in range(16):
                a = torch.unsqueeze(z[:, i], dim=1).to(device)
                b = torch.zeros(len(z[:, i]), c[i]).to(device).scatter_(1, a, 1)
                if i == 0:
                    x = b
                else:
                    x = torch.cat([x, b], 1)
        if client == 3:
            z = (x).long()
            c = [6, 8, 10, 9, 10, 3, 4, 5, 5, 3, 3, 3, 2,38,43,43,43]
            for i in range(17):
                a = torch.unsqueeze(z[:, i], dim=1).to(device)
                b = torch.zeros(len(z[:, i]), c[i]).to(device).scatter_(1, a, 1)
                if i == 0:
                    x = b
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out


class Client_LinearNet_2_em_census(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LinearNet_2_em_census, self).__init__()

        if client == 1:
            self.layer1 = nn.Sequential(nn.Linear(7, n_hidden_1),
                                        )
        if client == 2:
            self.layer1 = nn.Sequential(nn.Linear(260, n_hidden_1),
                                        )
        if client == 3:
            self.layer1 = nn.Sequential(nn.Linear(238, n_hidden_1),
                                        )

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    )

    def forward(self, x, client):
        if client == 2:
            z = (x).long()
            c = [9, 3, 2, 24, 15, 5, 10, 2, 3, 6, 8, 6,52,47,17,51]
            for i in range(16):
                a = torch.unsqueeze(z[:, i], dim=1).to(device)
                b = torch.zeros(len(z[:, i]), c[i]).to(device).scatter_(1, a, 1)
                if i == 0:
                    x = b
                else:
                    x = torch.cat([x, b], 1)
        if client == 3:
            z = (x).long()
            c = [6, 8, 10, 9, 10, 3, 4, 5, 5, 3, 3, 3, 2,38,43,43,43]
            for i in range(17):
                a = torch.unsqueeze(z[:, i], dim=1).to(device)
                b = torch.zeros(len(z[:, i]), c[i]).to(device).scatter_(1, a, 1)
                if i == 0:
                    x = b
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out

