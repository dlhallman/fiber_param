#IMPORTS
import argparse
import time
import torch
import torch.nn as nn
from tqdm import trange
import warnings
import random
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

#PATH
import sys
sys.path.append('./')
sys.path.append('../')

#SELF IMPORTS
# from lib.datasets import * 
# from lib.models.param import *
# from lib.utils.misc import set_outdir,set_seed, Recorder
# from lib.vis.model import *
# from lib.vis.modes import *
# from lib.vis.animate import *

from datasets import *
# from models import Encoder, odeFunc, NODE, HBNODE, Decoder
from param import *
from utilities import RunningAverageMeter, normal_kl
from utils.misc import set_outdir,set_seed, Recorder


# parser = argparse.ArgumentParser(prefix_chars='-+/',
#                   description='[PARAMTERIZED] PARAMETERIZED parameters.')
# data_parser = parser.add_argument_group('Data Parameters')
# data_parser.add_argument('--dataset', type=str, default='FIB',
#                   help='Dataset types: [EE].')
# data_parser.add_argument('--data_dir', type=str, default='./data/FIB.npz',
#                   help='Directory of data from cwd: sci.')
# data_parser.add_argument('--out_dir', type=str, default='./out/fib/',
#                   help='Directory of output from cwd: sci. SAVES TO JUSTIN/PODHBNODE/SRC/OUT')
# data_parser.add_argument('--modes', type = int, default = 4,
#                   help = 'POD reduction modes.')
# data_parser.add_argument('--tstart', type = int, default=0,
#                   help='Start time for reduction along time axis.')
# data_parser.add_argument('--tstop', type=int, default=302,
#                   help='Stop time for reduction along time axis.' )
# data_parser.add_argument('--batch_size', type=int, default=10,
#               help='Time index for validation data.' )
# data_parser.add_argument('--tr_ind', type=int, default=150,
#               help='Time index for data and label separation.' )
# data_parser.add_argument('--param_ind', type=int, default=70,
#               help='Param index for validation data. HOW MANY PARAMETERS FOR TRAINING'  )
# model_params = parser.add_argument_group('Model Parameters')
# model_params.add_argument('--model', type=str, default='HBNODE',
#                   help='Model choices - GHBNODE, HBNODE, NODE.')
# model_params.add_argument('--corr', type=int, default=-100,
#                   help='Skip gate input into soft max function.')
# train_params = parser.add_argument_group('Training Parameters')
# train_params.add_argument('--epochs', type=int, default=30,
#                   help='Training epochs.')
# train_params.add_argument('--layers', type=int, default=2,
#               help='Encoder Layers.')
# train_params.add_argument('--lr', type=float, default=0.01,
#                   help = 'Initial learning rate.')
# train_params.add_argument('--factor', type=float, default=0.99,
#                   help = 'Initial learning rate.')
# train_params.add_argument('--cooldown', type=int, default=0,
#                   help = 'Initial learning rate.')
# train_params.add_argument('--patience', type=int, default=5,
#                   help = 'Initial learning rate.')
# uq_params = parser.add_argument_group('Unique Parameters')
# uq_params.add_argument('--verbose', type=bool, default=False,
#               help='Display full NN and all plots.')
# uq_params.add_argument('--seed', type=int, default=0,
#                 help='Set initialization seed')
# uq_params.add_argument('--device', type=str, default='cpu',
#               help='Device argument for training.')

"""INPUT ARGUMETNS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='[NODE] NODE parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--data_dir', type=str, default='./data/FIB.npz',
                    help='Directory of data from cwd: sci.')
data_parser.add_argument('--out_dir', type=str, default='./out/fib/',
                    help='Directory for saving data.')
data_parser.add_argument('--tr_ind', type = int, default=150,
                    help='Time index for training data.')
# data_parser.add_argument('--val_ind', type=int, default=150,
#                     help='Time index for validation data.' )
data_parser.add_argument('--modes', type = int, default = 4,
                  help = 'POD reduction modes.')
data_parser.add_argument('--tstart', type = int, default=0,
                  help='Start time for reduction along time axis.')
data_parser.add_argument('--tstop', type=int, default=302,
                  help='Stop time for reduction along time axis.' )
data_parser.add_argument('--batch_size', type=int, default=10,
              help='Time index for validation data.' )
data_parser.add_argument('--param_ind', type=int, default=70,
              help='Param index for validation data. HOW MANY PARAMETERS FOR TRAINING'  )

data_parser.add_argument('--dataset', type=str, default='FIB',
                  help='Dataset types: [EE].')
""" MODEL """
model_parser = parser.add_argument_group('Model Parameters')
model_parser.add_argument('--model', type=str, default='NODE',
                    help='Dataset types: [NODE , HBNODE].')
model_parser.add_argument('--hidden_dim', type=int, default=6,
                    help = 'Size of latent dimension')
model_parser.add_argument('--layers_enc', type=int, default=4,
                help='Encoder Layers.')
model_parser.add_argument('--units_enc', type=int, default=10,
                    help='Encoder units.')
model_parser.add_argument('--layers_node', type=int, default=[12],
                nargs='+', help='NODE Layers.')
model_parser.add_argument('--units_dec', type=int, default=41,
                    help='Training iterations.')
model_parser.add_argument('--layers_dec', type=int, default=4,
                help='Encoder Layers.')

model_parser.add_argument('--corr', type=int, default=-100,
                  help='Skip gate input into soft max function.')

""" TRAIN """
train_parser = parser.add_argument_group('Training Parameters')
train_parser.add_argument('--lr', type=float, default=0.01,
                    help = 'Initial learning rate.')
train_parser.add_argument('--factor', type=float, default=0.99,
                    help = 'Factor for reducing learning rate.')
train_parser.add_argument('--epochs', type=int, default=100,
                    help='Training epochs.')
""" UNIQUE """
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--seed', type=int, default=1242,
                help='Set initialization seed')
uq_params.add_argument('--verbose', default=False, action='store_true',
                help='Display full NN and all plots.')
uq_params.add_argument('--device', type=str, default='cpu',
              help='Device argument for training.')


args, unknown = parser.parse_known_args()

# Load parameters
# Should probably go into datasets.py but I'm going to leave this here
# parameters = np.load('./data/FIB_param.npz')['arr_0']


if args.verbose:
    print('Parsed Arguments')
    for arg in vars(args):
        print('\t',arg, getattr(args, arg))

# """INITIALIZE"""
# set_seed(args.seed)
# set_outdir(args.out_dir, args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
""" SEEDIG """
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

#DATA LOADER
param = PARAM_DATASET(args)

# Check out an animation of the data - random parameters
samples = random.sample(range(0,len(param.data_init)),4)
mofo = param.data_init[[1,2,93,94],:,:]
# data_animation(mofo,args)
print('Done animating')


MODELS = {'NODE' : NMODEL(args),'HBNODE' : HBMODEL(args, res=True, cont=True), 'GHBNODE' : GHBMODEL(args, res=True, cont=True)}
# print('Generating ...\t Model: VAE '+args.model)
# obs_dim = param.train_data.shape[1]
# hidden_dim = args.hidden_dim

# MODELS = {'NODE' : NODE(df = odeFunc(hidden_dim,hidden_dim)),
#         'HBNODE' : HBNODE(odeFunc(hidden_dim,hidden_dim))}


#MODEL DIMENSIONS
assert args.model in MODELS
print('Generating ...\t Model: PARAMETER {}'.format(args.model))
model = MODELS[args.model].to(args.device)
if args.verbose:
    print(model.__str__())
    print('Number of Parameters: {}'.format(count_parameters(model)))

#LEARNING UTILITIES
gradrec = True
torch.manual_seed(0)
rec = Recorder()
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
loss_meter_t = RunningAverageMeter()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                factor=args.factor, patience=5, verbose=False, threshold=1e-5,
                threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)

# TRAINING
print('Training ... \t Iterations: {}'.format(args.epochs))
epochs = trange(1,args.epochs+1)
for epoch in epochs:

    rec['epoch'] = epoch
    batchsize = args.batch_size
    train_start_time = time.time()

    #SCHEDULER
    for param_group in optimizer.param_groups:
        rec['lr'] = param_group['lr']
    scheduler.step(metrics=loss_meter_t.avg)

    #BATCHING
    for b_n in range(0, param.train_data.shape[1], batchsize):
        # model.cell.nfe = 0
        model.nfe = 0
        predict = model(param.train_times[:, b_n:b_n + batchsize], param.train_data[:, b_n:b_n + batchsize], param.param_train[b_n:b_n+batchsize])
        loss = criteria(predict, param.train_label[:, b_n:b_n + batchsize])
        loss_meter_t.update(loss.item())
        rec['tr_loss'] = loss
        rec['forward_nfe'] = model.cell.nfe
        epochs.set_description('loss:{:.3f}'.format(loss))

        #BACKPROP
        if gradrec is not None:
            lossf = criteria(predict[-1], param.train_label[-1, b_n:b_n + batchsize])
            lossf.backward(retain_graph=True)
            vals = model.ode_rnn.h_rnn
            for i in range(len(vals)):
                grad = vals[i].grad
                rec['grad_{}'.format(i)] = 0 if grad is None else torch.norm(grad)
            model.zero_grad()
        model.cell.nfe = 0
        loss.backward()
        optimizer.step()
        rec['backward_nfe'] = model.cell.nfe
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    rec['train_time'] = time.time() - train_start_time

    #VALIDATION
    if epoch == 0 or (epoch + 1) % 1 == 0:
        model.cell.nfe = 0
        predict = model(param.valid_times, param.valid_data, param.param_valid)
        vloss = criteria(predict, param.valid_label)
        rec['val_nfe'] = model.cell.nfe
        rec['val_loss'] = vloss

    #TEST
#    if epoch == 0 or (epoch + 1) % 5 == 0:
#        model.cell.nfe = 0
#        predict = model(param.eval_times, param.eval_data)
#        sloss = criteria(predict, param.eval_label)
#        sloss = sloss.detach().cpu().numpy()
#        rec['ts_nfe'] = model.cell.nfe
#        rec['ts_loss'] = sloss

    #OUTPUT
    rec.capture(verbose=False)
    if (epoch + 1) % 5 == 0:
        torch.save(model, args.out_dir+'/pth/{}.mdl'.format(args.model))
        rec.writecsv(args.out_dir+'/pth/{}.csv'.format(args.model))
print("Generating Output ... ")
rec_file = args.out_dir+ './pth/'+args.model+'.csv'
rec.writecsv(rec_file)
args.model = str('param_'+args.model).lower()
tr_pred= model(param.train_times, param.train_data).cpu().detach()
tr_pred = tr_pred[-param.label_len:]

val_pred = model(param.valid_times, param.valid_data).cpu().detach().numpy()
val_pred = val_pred[-param.label_len:]

trained = np.vstack((param.train_data[:args.tr_ind],tr_pred))
validated = np.vstack((param.valid_data[:args.tr_ind],val_pred))

trained_true = np.vstack((param.train_data[:args.tr_ind],param.train_label[-param.label_len:]))
validated_true = np.vstack((param.valid_data[:args.tr_ind],param.valid_label[-param.label_len:]))

times = np.arange(args.tstart,args.tstop)

data = np.hstack((trained,validated))*param.std_data+param.mean_data
data_true = np.hstack((trained_true,validated_true))*param.std_data+param.mean_data

#DATA PLOTS
verts = [args.tstart+args.tr_ind]
true = np.moveaxis(param.data.copy(),0,1)
mode_prediction(data[:,args.param_ind+2,:4],data_true[:,args.param_ind+2,:4],times,verts,args,'_val')
mode_prediction(data[:,0,:4],data_true[:,0,:4],times,verts,args)
#%%
val_recon = pod_mode_to_true(param.pod_dataset,normalized,args)
data_reconstruct(val_recon,-1,args)
data_animation(val_recon,args)
#%%
#MODEL PLOTS
plot_loss(rec_file, args)
plot_nfe(rec_file,'forward_nfe', args)
plot_adjGrad(rec_file, args)
plot_stiff(rec_file, args)
