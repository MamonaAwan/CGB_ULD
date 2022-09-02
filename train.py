import glob, os

import torch, time, numpy as np
from torch.utils.data import DataLoader
from shutil import copy2

from model_fan.model import model as mymodel
from database.databases import Full_DB
from utils_fan.utils import *from utils_fan.train_options import Options
from utils_fan.helper_functions import store_landmark_features
from utils_gcn import create_logger
from vegcn.gcn_main import gcn_main


def main():
    # parse args
    global args
    args = Options().args
    
    # copy all files from experiment
    cwd = os.getcwd()
    for ff in glob.glob("*.py"):
        copy2(os.path.join(cwd,ff), os.path.join(args.folder,'code'))

    # initialise seeds
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    np.random.seed(3407)


    # parameters
    sigma = float(args.s)
    temperature = float(args.t)
    gradclip = int(args.gc)
    npts = int(args.npts)
    bSize = int(args.bSize)
    angle = float(args.angle)
    flip = eval(str(args.flip))
    tight = int(args.tight)

    # load the losses class
    loss_cf = loss_to_use(args)

    model = mymodel(sigma=sigma, temperature=temperature, 
                    gradclip=gradclip,  npts=npts, 
                    option=args.option, size=args.size, 
                    path_to_check=args.checkpoint, for_clustering = True)
    
    losskeys = list(model.loss.keys())
            
    # prepare average meters
    global meters
    meterskey = ['batch_time'] 
    meters = dict([(key,AverageMeter()) for key in meterskey])
    meters['losses'] = dict([(key,AverageMeter()) for key in losskeys])
     
  
    # define data
    video_dataset = Full_DB(path=args.data_path,sigma=sigma,size=args.size,flip=flip,angle=angle,tight=tight, db=args.db)
    videoloader = DataLoader(video_dataset, batch_size=bSize, shuffle=True, num_workers=int(args.num_workers), pin_memory=True)
    print('Number of workers is {:d}, and bSize is {:d}'.format(int(args.num_workers),bSize))
       
    # define optimizers
    lr_fan = args.lr_fan
    lr_gan = args.lr_gan
    print('Using learning rate {} for FAN, and {} for GAN'.format(lr_fan,lr_gan))
    optimizerFAN = torch.optim.Adam(model.FAN.parameters(), lr=lr_fan, betas=(0, 0.9), weight_decay=5*1e-4)
    schedulerFAN = torch.optim.lr_scheduler.StepLR(optimizerFAN, step_size=args.step_size, gamma=args.gamma)
    optimizerGEN = torch.optim.Adam(model.GEN.parameters(), lr=lr_gan, betas=(0, 0.9), weight_decay=5*1e-4)
    schedulerGEN = torch.optim.lr_scheduler.StepLR(optimizerGEN, step_size=args.step_size, gamma=args.gamma)
    myoptimizers = {'FAN' : optimizerFAN, 'GEN' : optimizerGEN}

    # path to save models and images
    path_to_model = os.path.join(args.folder,args.file)

    # gcn logger 
    gcn_logger = create_logger(log_file= os.path.join(args.folder, 'gcn_log.txt'))
    
    # train model and extract representations (of landmarks)
    initial = True
    n_epochs = 5
    tot_epochs = 145
    astart = 10
    # n_epochs = 1
    # tot_epochs = 10
    # astart = 1
    rounds = int(tot_epochs/n_epochs)-1
    ne_epoches = [astart+(r*n_epochs) for r in range(0, rounds)]
    ne_epoches = [0] + ne_epoches

    for r in range(0,int(rounds)):
        lo_repr_all = []
        for epoch in range(ne_epoches[r], ne_epoches[r+1], 1):
            lo_rp = train_epoch(videoloader, model, myoptimizers, epoch, bSize, loss_cf, initial= initial)
            lo_repr_all.append(lo_rp)
            schedulerFAN.step()
            schedulerGEN.step()
            if epoch == tot_epochs-1 or epoch % 10 == 0 and epoch != 0:
                model._save(path_to_model,epoch)

        lo_repr_all = torch.cat(lo_repr_all, dim = 0)

        # run gcn except for last round
        if r != rounds-1:
            store_landmark_features(lo_repr_all, d = 256)
            gcn_logger.info('-------------------------------------------')
            gcn_logger.info('Total Landmark Fts : {}'.format(lo_repr_all.shape))
            gcn_logger.info('Processing Round {}'.format(r))
            predictions_dir = gcn_main(args.folder, gcn_logger)
            del lo_repr_all 
        
        # engage the full model after initial round
        if r == 0:
            initial = False
            model.for_clustering = False
            model._set_pred_dir(predictions_dir)
            model._engage_full(loss_cf=loss_cf)
            losskeys = list(model.loss.keys())
            meters['losses'] = dict([(key,AverageMeter()) for key in losskeys])
        else:
            model.new_round = True


    

def train_epoch(dataloader, model, myoptimizers, epoch, bSize, loss_cf, initial):
    
    itervideo = iter(dataloader)
    log_epoch = {}
    end = time.time()

    lo_repr_fts = []
    
    for i in range(0,2500): 
    
        # - get data
        all_data = next(itervideo,None) 
        if all_data is None:
            itervideo = iter(dataloader)
            all_data = next(itervideo, None)
        elif all_data['Im'].shape[0] < bSize:
            itervideo = iter(dataloader)
            all_data = next(itervideo, None)
        
        # - set batch
        model._set_batch(all_data)
        
        # - forward
        output = model.forward()

        # - update parameters
        myoptimizers['GEN'].step()
        myoptimizers['FAN'].step()

        if initial:
            meters['losses']['rec'].update(model.loss['rec'].item(), bSize)

        else:
            if loss_cf.ada_rec:
                meters['losses']['ada_rec'].update(model.loss['ada_rec'].item(), bSize)
            
        
        if i % 250 == 0:
            # - storing the representations 
            lo_repr_fts.append(output['Repr'].cpu())
            
        
        log_epoch[i] = model.loss       
        meters['batch_time'].update(time.time()-end)
        end = time.time()
        if i % args.print_freq == 0:
            mystr = 'Epoch [{}][{}/{}] '.format(epoch, i, len(dataloader))
            mystr += 'Time {:.2f} ({:.2f}) '.format(meters['batch_time'].val , meters['batch_time'].avg )
            mystr += ' '.join(['Loss: {:s} {:.3f} ({:.3f}) '.format(k, meters['losses'][k].val , meters['losses'][k].avg ) for k in meters['losses'].keys()])
            print( mystr )
            with open(args.folder + '/args_' + args.file[0:-8] + 'log.txt','a') as f: 
                print( mystr , file=f)


    lo_repr_fts= torch.cat(lo_repr_fts, dim= 0)
    
    return lo_repr_fts



class loss_to_use():
    '''
    set loss to use.
    '''
    def __init__(self, args):
        self.ada_rec = args.ada_rec
        





if __name__ == '__main__':
    main()


