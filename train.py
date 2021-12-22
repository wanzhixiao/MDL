import torch
import numpy as np
import argparse
from util import *
from model import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from collections import defaultdict
import math
import copy,time


parser = argparse.ArgumentParser()
#parameter of dataset
parser.add_argument('--len_trend',type=int,default=1,help='length of trend data')
parser.add_argument('--len_period',type=int,default=1,help='length of period data')
parser.add_argument('--len_closeness',type=int,default=3,help='length of closeness data')
parser.add_argument('--train_prop',type=float,default=0.8,help='proportion of training set')
parser.add_argument('--val_prop',type=float,default=0.1,help='proportion of validation set')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--height',type=int,default=16,help='input flow image height')
parser.add_argument('--width',type=int,default=16,help='input flow image width')
parser.add_argument('--external_dim',type=int,default=28,help='external factor dimension')
parser.add_argument('--embed_dim',type=int,default=16,help='edge channel embedding dimension')

#parameter of training
parser.add_argument('--epochs',type=int,default=100,help='training epochs')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--seed',type=int,default=99,help='running seed')
parser.add_argument('--save_folder',type=str,default='./result',help='result dir')
parser.add_argument('--device',type=str,default='cuda:0',help='cuda device')
parser.add_argument('--max_grad_norm',type=int,default=5,help='max gradient norm for gradient clip')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
#parameter of model

args = parser.parse_args()


def train(model,
          dataloaders,
          optimizer,
          epochs,
          folder,
          external_dim,
          early_stop_steps = 10,
          device = 'cpu',
          max_grad_norm = None):

    #1. save path
    save_path = os.path.join(folder,'model', 'best_model.pkl')
    tensorboard_folder = os.path.join(folder,'tensorboard')

    if os.path.exists(save_path):
        print('path exists')
        save_dict = torch.load(save_path)
        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1

        #move the load parameter tensor to cuda, for optimizer.step()
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    else:
        print('path not exists')
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch  = 0


    if not os.path.exists(tensorboard_folder):
        os.makedirs(tensorboard_folder)

    writer = SummaryWriter(tensorboard_folder)
    since = time.perf_counter()

    phases = ['train', 'validate']
    model = model.to(device)
    # optimizer
    print(model)

    #2. train model
    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):
            running_loss, running_metrics = defaultdict(float), dict()

            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                tqdm_loader = tqdm(enumerate(dataloaders[phase]))

                steps, flow_pred, flow_targets = 0, list(), list()
                od_flow_pred, od_flow_targets = list(), list()

                for step, data in tqdm_loader:
                    flow, od_flow, flow_label, od_flow_label = data['flow'], data['od_flow'],\
                                                               data['flow_label'], data['od_flow_label']

                    flow_targets.append(flow_label.numpy().squeeze(1))
                    od_flow_targets.append(od_flow_label.numpy().squeeze(1))

                    #todo using external factor to inplace random noise
                    external_input = torch.randn(flow.shape[0], external_dim)

                    with torch.no_grad():
                        flow = flow.to(device)
                        od_flow = od_flow.to(device)
                        flow_label = flow_label.to(device)
                        od_flow_label = od_flow_label.to(device)
                        external_input = external_input.to(device)

                    x_flow = [flow[:,:args.len_trend,...],flow[:,args.len_trend:args.len_trend+args.len_period,...],flow[:,-args.len_closeness:,...]]
                    x_od_flow = [od_flow[:,:args.len_trend,...],od_flow[:,args.len_trend:args.len_trend+args.len_period,...],od_flow[:,-args.len_closeness:,...]]

                    with torch.set_grad_enabled(phase == 'train'):
                        loss_all, node_pred, edge_pred = model.multask_loss(x_flow, x_od_flow, external_input,\
                                                                            flow_label, od_flow_label)

                        if phase == "train":
                            optimizer.zero_grad()
                            loss_all.backward()

                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        flow_pred.append(node_pred.cpu().numpy())
                        od_flow_pred.append(edge_pred.cpu().numpy())

                    running_loss[phase] += loss_all * len(flow)
                    steps += len(flow)

                    torch.cuda.empty_cache()

                running_metrics[phase] = evaluate(np.concatenate(flow_pred),np.concatenate(flow_targets),\
                                                 np.concatenate(od_flow_pred), np.concatenate(od_flow_targets),
                                                dataloaders['flow_scaler'], dataloaders['od_flow_scaler'])
                #for select model
                if phase == 'validate':
                    if running_loss['validate'] <= best_val_loss or math.isnan(running_loss['validate']):
                        best_val_loss = running_loss['validate']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        save_model(save_path, **save_dict)
                        print(f'Better model at epoch {epoch} recorded.')
                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')

            for phase in phases:
                for key,val in running_metrics[phase].items():
                    writer.add_scalars(f'{phase}', {f'{key}': val}, global_step=epoch)
                    writer.add_scalars('Loss', {
                        f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
                                       global_step=epoch)
                    print(f'epoch:{epoch},{phase} loss:{running_loss[phase]},{key}:{val}')
    except:
        writer.close()
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')

    writer.close()


def test_model(folder,
               model,
               dataloaders,
               device,
               external_dim):

    save_path = os.path.join(folder, 'model', 'best_model.pkl')
    save_dict = torch.load(save_path)
    model.load_state_dict(save_dict['model_state_dict'])
    model = model.to(device)

    # model.eval()
    steps, flow_pred, flow_targets = 0, list(), list()
    od_flow_pred, od_flow_targets = list(), list()

    tqdm_loader = tqdm(enumerate(dataloaders['test']))

    for step, data in tqdm_loader:
        flow, od_flow, flow_label, od_flow_label = data['flow'], data['od_flow'], \
                                                   data['flow_label'], data['od_flow_label']

        flow_targets.append(flow_label.numpy().squeeze(1))
        od_flow_targets.append(od_flow_label.numpy().squeeze(1))

        # todo using external factor to inplace random noise
        external_input = torch.randn(flow.shape[0], external_dim)

        with torch.no_grad():
            flow = flow.to(device)
            od_flow = od_flow.to(device)
            flow_label = flow_label.to(device)
            od_flow_label = od_flow_label.to(device)
            external_input = external_input.to(device)

            x_flow = [flow[:, :args.len_trend, ...], flow[:, args.len_trend:args.len_trend + args.len_period, ...],
                      flow[:, -args.len_closeness:, ...]]
            x_od_flow = [od_flow[:, :args.len_trend, ...], od_flow[:, args.len_trend:args.len_trend + args.len_period, ...],
                         od_flow[:, -args.len_closeness:, ...]]

            _, node_pred, edge_pred = model.multask_loss(x_flow, x_od_flow, external_input, \
                                                                flow_label, od_flow_label)

            flow_pred.append(node_pred.cpu().numpy())
            od_flow_pred.append(edge_pred.cpu().numpy())

    flow_targets, od_flow_targets = np.concatenate(flow_targets,axis=0), np.concatenate(od_flow_targets,axis=0)
    flow_pred, od_flow_pred = np.concatenate(flow_pred, axis=0), np.concatenate(od_flow_pred, axis=0)

    scores = evaluate(flow_pred, flow_targets, \
             od_flow_pred, od_flow_targets,
             dataloaders['flow_scaler'], dataloaders['od_flow_scaler'])

    print('test results:')
    print(json.dumps(scores,cls=MyEncoder, indent=4))

    with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
        json.dump(scores, f,cls=MyEncoder, indent=4)

    np.savez(os.path.join(folder, 'test-results.npz'), flow_pred=flow_pred, flow_targets=flow_targets, \
             od_flow_pred=od_flow_pred, od_flow_targets=od_flow_targets)

def main():
    # 1.set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    #2. load data
    dataloader_s = get_dataloader(args.len_trend, args.len_period, args.len_closeness,\
                                   train_prop=args.train_prop,val_prop=args.val_prop, batch_size=args.batch_size)

    #3. construct model
    height, width =  args.height, args.width
    node_channels = 2
    edge_channels = 2*height * width
    model = MDL(node_conf = (args.len_closeness,node_channels, height, width),
                node_tconf = (args.len_trend,node_channels, height, width),
                node_pconf = (args.len_period, node_channels, height, width),
                edge_conf = (args.len_closeness, edge_channels, height, width),
                edge_tconf = (args.len_trend, edge_channels, height, width),
                edge_pconf = (args.len_period, edge_channels, height, width),
                external_dim = args.external_dim,
                embed_dim = args.embed_dim)

    #4. optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    #5. train
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train(model = model,
        dataloaders = dataloader_s,
        optimizer = optimizer,
        epochs = args.epochs,
        folder = args.save_folder,
        external_dim = args.external_dim,
        early_stop_steps=10,
        device=device,
        max_grad_norm=args.max_grad_norm)


    #6. test
    test_model(folder = args.save_folder,
               model = model,
               dataloaders = dataloader_s,
               device = device,
               external_dim = args.external_dim)

if __name__ == '__main__':
    main()