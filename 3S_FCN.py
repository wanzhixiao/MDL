import torch
import torch.nn as nn
import copy


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes,stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        #conv->bn->relu => conv->relu->bn
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        return out

class STModule(nn.Module):
    def __init__(self,block,conf,nb_residual_unit=2,edge_net=False,k=64):
        super(STModule, self).__init__()
        len_seq, nb_flow, map_height, map_width = conf
        if not edge_net:
            self.conv1 =nn.Conv2d(nb_flow * len_seq, 64, kernel_size=(3, 3), stride=1, padding=1)
            self.layer1 = self._make_layer(block, inplanes=64, planes=64, repetitions=nb_residual_unit)
            self.layer2 = self._make_layer(block, inplanes=64, planes=64, repetitions=nb_residual_unit)
            self.layer3 = self._make_layer(block, inplanes=64, planes=64, repetitions=nb_residual_unit)
            self.conv2 = nn.Conv2d(64, 2, kernel_size=(3, 3), stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(k, 128, kernel_size=(3, 3), stride=1, padding=1)
            self.layer1 = self._make_layer(block, inplanes=128, planes=128, repetitions=nb_residual_unit)
            self.layer2 = self._make_layer(block, inplanes=128, planes=128, repetitions=nb_residual_unit)
            self.layer3 = self._make_layer(block, inplanes=128, planes=128, repetitions=nb_residual_unit)
            self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)



    def _make_layer(self,block,inplanes=64,planes=64,repetitions=3,stride=1):
        layers = []
        for i in range(repetitions):
            layers.append(block(inplanes, planes,stride=stride))
        return nn.Sequential(*layers)

    def forward(self,x):
        print(x.shape)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.conv2(out)
        return out


class FCN(nn.Module):
    def __init__(self,c_conf=(3, 2, 8, 8), p_conf=(1, 2, 8, 8), t_conf=(1, 2, 8, 8),nb_residual_unit=2,
                 edge_net=False,block=BasicBlock):
        super(FCN, self).__init__()

        #closseness,period,trend module

        #external
        self.w_c = nn.Parameter(torch.randn(1))
        self.w_p = nn.Parameter(torch.randn(1))
        self.w_t = nn.Parameter(torch.randn(1))
        self.relu = nn.ReLU(inplace=True)

        self.edge_net = edge_net
        if self.edge_net:
            #formular 14,reduction channels from 2N to k
            self.k = 128
            self.c_module = STModule(block, conf=c_conf, nb_residual_unit=nb_residual_unit, edge_net=edge_net,k=128)
            self.p_module = STModule(block, conf=p_conf, nb_residual_unit=nb_residual_unit, edge_net=edge_net,k=128)
            self.t_module = STModule(block, conf=t_conf, nb_residual_unit=nb_residual_unit, edge_net=edge_net,k=128)

            self.embed_c = nn.Linear(c_conf[1]*c_conf[2]*c_conf[3],self.k*c_conf[2]*c_conf[3],bias=True)
            self.embed_p = nn.Linear(p_conf[1]*p_conf[2]*p_conf[3],self.k*p_conf[2]*p_conf[3],bias=True)
            self.embed_t = nn.Linear(t_conf[1]*t_conf[2]*t_conf[3],self.k*t_conf[2]*t_conf[3],bias=True)
        else:
            self.c_module = STModule(block, conf=c_conf, nb_residual_unit=nb_residual_unit, edge_net=edge_net)
            self.p_module = STModule(block, conf=p_conf, nb_residual_unit=nb_residual_unit, edge_net=edge_net)
            self.t_module = STModule(block, conf=t_conf, nb_residual_unit=nb_residual_unit, edge_net=edge_net)

    def forward(self,X_train):
        X_c,X_p,X_t = X_train[0],X_train[1],X_train[2]
        if self.edge_net:
            h,w = X_c.shape[-2],X_c.shape[-1]
            X_c = X_c.reshape(X_c.shape[0],-1)
            X_p = X_p.reshape(X_p.shape[0],-1)
            X_t = X_t.reshape(X_t.shape[0],-1)

            X_c = self.embed_c(X_c)
            X_p = self.embed_p(X_p)
            X_t = self.embed_t(X_t)

            X_c = X_c.reshape(X_c.shape[0],-1,h,w)
            X_p = X_c.reshape(X_p.shape[0],-1,h,w)
            X_t = X_c.reshape(X_t.shape[0],-1,h,w)

        print('fff',X_c.shape)

        c_out = self.c_module(X_c)
        p_out = self.p_module(X_p)
        t_out = self.t_module(X_t)
        #parameter matrix fusion
        out = torch.add(self.w_c*c_out,self.w_p*p_out)
        out = torch.add(out,self.w_t*t_out)
        return out



class MDL(nn.Module):
    def __init__(self,node_conf,node_tconf,node_pconf,
                 edge_conf,edge_tconf,edge_pconf,external_dim=28):
        super(MDL, self).__init__()

        self.node_net = FCN(c_conf=node_conf, p_conf=node_tconf, t_conf=node_pconf,nb_residual_unit=2,edge_net=False)
        self.edge_net = FCN(c_conf=edge_conf, p_conf=edge_tconf, t_conf=edge_pconf,nb_residual_unit=2,edge_net=True)

        self.node_conv = nn.Conv2d(in_channels=64+2,out_channels=2,kernel_size=3,stride=1,padding=1)
        self.edge_conv = nn.Conv2d(in_channels=64+2,out_channels=8*edge_conf[-2]*edge_conf[-1],kernel_size=3,stride=1,padding=1)
        self.external_dim = external_dim

        self.fc1 = nn.Linear(in_features=external_dim,out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=2*8*8)
        self.fc3 = nn.Linear(in_features=10, out_features=512*8*8)

        # self.w_node = nn.Parameter(torch.randn(1).cuda())
        # self.w_edge = nn.Parameter(torch.randn(1).cuda())
        # self.w_mdl = nn.Parameter(torch.randn(1).cuda())
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.relu = nn.ReLU(inplace=True)
        self.w_node = 1
        self.w_edge = 1
        self.w_mdl = 0.0005

    def fusion_external(self, X_ext, flow, planel,edge_net=False):
        if self.external_dim != None and self.external_dim > 0:
            external_out = self.fc1(X_ext)
            external_out = self.relu(external_out)
            if not edge_net:
                external_out = self.fc2(external_out)
            else:
                external_out = self.fc3(external_out)
            external_out = self.relu(external_out)
            external_out = external_out.reshape(external_out.shape[0], planel, 8, 8)
        else:
            print('external_dim:', self.external_dim)
        # gating
        external_out = torch.sigmoid(external_out)

        print(external_out.shape,flow.shape)

        out = external_out * flow
        out = torch.tanh(out)

        return out

    def forword(self,X,M,X_ext):
        '''
        :param X: the node net input,shape (b,c,h,w)
        :param M: the edge net input,shape (b,c',h,w), c' = h*w*2
        :param X_ext: external imformation,shape(b,f),f is external dim
        :return:
        '''
        node_flow = self.node_net(X)
        edge_flow = self.edge_net(M)
        #todo cross concat
        concat_flow = torch.cat([node_flow,edge_flow],dim=1)

        print(concat_flow.shape)
        #to fusion external
        node_out = self.node_conv(concat_flow)
        edge_out = self.edge_conv(concat_flow)

        node_out = self.fusion_external(X_ext,node_out,planel=2)
        edge_out = self.fusion_external(X_ext,edge_out,planel=8*M[0].shape[-2]*M[0].shape[-1],edge_net=True)

        return node_out,edge_out


    def multask_loss(self,X,M,X_ext,X_gt,M_gt):

        node_pred, edge_pred = self.forword(X,M,X_ext)

        # indication matrix
        P_node = copy.deepcopy(X_gt)
        P_node[P_node > 0] = 1

        Q_edge = copy.deepcopy(M_gt)
        Q_edge[Q_edge > 0] = 1

        print(M_gt.shape,edge_pred.shape)

        #node loss
        node_loss = torch.mul(torch.sum(P_node*(X_gt-node_pred)*(X_gt-node_pred)),self.w_node)
        edge_loss = torch.mul(torch.sum(Q_edge*(M_gt-edge_pred)*(M_gt-edge_pred)),self.w_edge)

        #mdl loss
        #out flow - outgoing transitions
        out_loss = X_gt[:,0,:,:] - torch.sum(M_gt[:,:M_gt.shape[-1],:,:],dim=1)
        # in flow - incoming transitions
        in_loss = X_gt[:,1,:,:] - torch.sum(M_gt[:,M_gt.shape[-1]:,:,:],dim=1)
        mdl_loss = torch.mul(torch.sum(out_loss*out_loss+in_loss*in_loss),self.w_mdl)

        loss_all = node_loss+edge_loss+mdl_loss

        node_rmse = torch.sqrt(self.mse(node_pred,X_gt))
        edge_rmse = torch.sqrt(self.mse(edge_pred,M_gt))

        node_mae = self.mae(node_pred,X_gt)
        edge_mae = self.mae(edge_pred,M_gt)

        return loss_all,node_rmse,edge_rmse,node_mae,edge_mae



if __name__ == '__main__':

    #external input,28 is feature dim
    X_ext = torch.rand((1,28))
    # node input,(b,c,h,w)
    X_c = torch.rand((1,3*2,8,8))
    X_p = torch.rand((1,1*2,8,8))
    X_t = torch.rand((1,1*2,8,8))

    X_node = [X_c,X_p,X_t]
    # edge input
    M_c = torch.rand((1,3*512,8,8))
    M_p = torch.rand((1,1*512,8,8))
    M_t = torch.rand((1,1*512,8,8))

    M_edge = [M_c,M_p,M_t]


    #label
    X_gt = torch.rand((1,2,8,8))
    M_gt = torch.rand((1,512,8,8))

    model = MDL(node_conf=(3,2,8,8),node_tconf=(1,2,8,8),node_pconf=(1,2,8,8),
                 edge_conf=(3,512*3,8,8),edge_tconf=(1,512,8,8),edge_pconf=(1,512,8,8))

    loss_all, node_rmse, edge_rmse,\
    node_mae, edge_mae = model.multask_loss(X_node,M_edge,X_ext,X_gt,M_gt)

    print(loss_all, node_rmse, edge_rmse)
    print(node_mae, edge_mae)

    loss_all.backward()