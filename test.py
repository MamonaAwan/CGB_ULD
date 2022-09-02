import os
import torch, numpy as np, math
import argparse
from torch.utils.data import DataLoader

from model_fan.MTFAN import FAN, GeoDistill
from database.databases import Full_DB

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-f','--folder', default='', type=str, metavar='PATH', help='folder')
parser.add_argument('-d','--db', default='MAFL', type=str, metavar='PATH', help='db')
parser.add_argument('-e', '--epoch', help='Epoch to test')
parser.add_argument('-t','--tight', default=70, type=int, help='tight')
parser.add_argument('--data_path', default='', help='Path to the data')



def main():
    # input parameters
    args = parser.parse_args()
    folder = args.folder
    epoch = args.epoch
    db = args.db  
    tight = args.tight
    data = extractdata(folder, epoch, db, tight, npoints=10, data_path=args.data_path)
    reg_factor = 0.01
    e = str(args.epoch)
    print('Doing epoch {}'.format(e))
    
    dbtrain = db + '-train'
    dbtest = db + '-test'
    Ytr = data[dbtrain][e]['Gtr']
    Ytr = Ytr.reshape(Ytr.shape[0],-1)/4
    Xtr = data[dbtrain][e]['Ptr']
    Xtr = Xtr.reshape(Xtr.shape[0],-1)
    Ytest = data[dbtest][e]['Gtr']
    Ytest = Ytest.reshape(Ytest.shape[0],-1)/4
    Xtest = data[dbtest][e]['Ptr']
    Xtest = Xtest.reshape(Xtest.shape[0],-1)

    m_fwd_errors = compute_errors(Xtr,Ytr,Xtest,Ytest,reg_factor,db,True) 
    m_bwd_errors = compute_errors(Ytr,Xtr,Ytest,Xtest,reg_factor,db,False)
    print('mean fwd error= ', m_fwd_errors)
    print('mean bwd error= ', m_bwd_errors)
    
   
def compute_errors(Xtr,Ytr,Xtest,Ytest,reg_factor,db,fwd=True):
    npts = 10
    n = [1,5,10,100,500,1000,5000,Xtr.shape[0]]
    nrepeats = 10
    all_errors = np.zeros((len(n),nrepeats))
    for tmp_idx in range(0,len(n)):
        for j in range(0,nrepeats):
            idx = np.random.permutation((range(0,Xtr.shape[0])))[0:n[tmp_idx]+1]
            R, X0, Y0 = train_regressor(Xtr[idx,:], Ytr[idx,:], reg_factor)
            err = np.zeros((Xtest.shape[0]))
            for i in range(0,Xtest.shape[0]):
                x = Xtest[i,:]
                y = Ytest[i,:]
                if fwd:
                    x = fit_regressor(R,x,X0,Y0)
                    err[i] = NMSE( y.reshape(-1,2), x, db)
                else:
                    b = x
                    x = fit_regressor(R,x,X0,Y0)
                    y = y.reshape(-1,2)
                    if db == 'MAFL' or db == 'AFLW': 
                        iod = compute_iod(b.reshape(-1,2))
                        err[i] = np.sum(np.sqrt(np.sum((x-y)**2,1)))/(iod*npts)
                    else:
                        normd = getnorm(b.reshape(-1,2))
                        err[i] = np.sum(np.sqrt(np.sum((x-y)**2,1))/normd)/(npts)
            all_errors[tmp_idx,j] = np.mean(err)
    return all_errors.mean(axis=1)[-1]        

def NMSE(landmarks_gt, landmarks_regressed, db):
    lm = landmarks_gt
    if len(landmarks_gt.shape) == 2:
        landmarks_gt = landmarks_gt.reshape(1,-1,2)
    if len(landmarks_regressed.shape) == 2:
        landmarks_regressed = landmarks_regressed.reshape(1,-1,2)
    if landmarks_gt.shape[1] == 5 or landmarks_gt.shape[1] == 10 or landmarks_gt.shape[1] == 9:
        eyes = landmarks_gt[:, :2, :]  
    else:
        eyes = landmarks_gt[:,[36,45],:]
    occular_distances = np.sqrt(np.sum((eyes[:, 0, :] - eyes[:, 1, :])**2, axis=-1))
    distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed)**2, axis=-1))
    if db == 'MAFL' or db == 'AFLW':
        mean_error = np.mean(distances / occular_distances[:, None])
    else:
        normd = getnorm(lm)
        mean_error = np.mean(distances)/normd
    return mean_error


def getnorm(landmarks_gt):
    h=np.max(landmarks_gt[:,1])-np.min(landmarks_gt[:,1])
    w=np.max(landmarks_gt[:,0])-np.min(landmarks_gt[:,0])
    normd =math.sqrt(h*w)
    return normd


def train_regressor(X,Y,l,size=64):
    center = size/2
    Xtmp = X/center - 0.5
    X0 = Xtmp.mean(axis=0, keepdims=True)
    Xtmp = Xtmp - np.ones((Xtmp.shape[0],1)) @ X0.reshape(1,-1)
    C = Xtmp.transpose() @ Xtmp
    Ytmp = Y/center - 0.5
    Y0 = Ytmp.mean(axis=0, keepdims=True)
    Ytmp = Ytmp - np.ones((Ytmp.shape[0],1)) @ Y0.reshape(1,-1)
    R = ( Ytmp.transpose() @ Xtmp ) @ np.linalg.inv( C + l*(C.max()+1e-12)*np.eye(Xtmp.shape[1])) 
    return R, X0, Y0

def fit_regressor(R,x,X0,Y0, size=64):
    center = size/2
    x = (R @ (x/center - 0.5 - X0).transpose()).reshape(-1,2) + Y0.reshape(-1,2)
    x = (x + 0.5)*center
    return x

def compute_iod(y):
    if y.shape[0] == 5:
        iod = np.sqrt( (y[0,0] - y[1,0])**2 + (y[0,1] - y[1,1])**2 )
    else:
        iod = np.sqrt( (y[36,0] - y[45,0])**2 + (y[36,1] - y[45,1])**2 )
    return iod



def loadnet(npoints=10,path_to_model=None):
    net = FAN(1,n_points=npoints).to('cuda')
    assert path_to_model is not None, 'Specified Model not Found!'
    net_dict = net.state_dict()
    pretrained_dict = torch.load(path_to_model, map_location='cuda')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrained_dict[k].shape == net_dict[k].shape}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict, strict=True)

    return net.to('cuda')

def getdata(loader, BOT, net):
    preds = []
    gths = []
    with torch.no_grad():
        for sample in loader:
            img = sample['Im']
            pts = sample['pts']
            _,out = BOT(net(img.cuda())[0])
            preds.append(out.cpu().detach())
            gths.append(pts)
    return np.concatenate(preds), np.concatenate(gths)

def get_truncated_data(loader, BOT, net, max):
    preds = []
    gths = []
    count = 1
    with torch.no_grad():
        for sample in loader:
            img = sample['Im']
            pts = sample['pts']
            _,out = BOT(net(img.cuda())[0])
            preds.append(out.cpu().detach())
            gths.append(pts)
            count+=1
            if count==max:
                break
    return np.concatenate(preds), np.concatenate(gths)


def extractdata(folder, epoch, db, tight, npoints, data_path):
    
    path_to_model = '{}/model_{}.fan.pth'.format(folder,epoch)
    net = loadnet(npoints,path_to_model)
    BOT = GeoDistill(sigma=0.5, temperature=0.1).to('cuda')

    dbs = [str(db + '-train'), str(db +'-test')]

    database = Full_DB(path=data_path,size=128,flip=False,angle=0.0,tight=tight or 64, db=dbs[0], affine=True)
    num_workers = 12 
    dbloader = DataLoader(database, batch_size=30, shuffle=False, num_workers=num_workers, pin_memory=False)
    # extract data        
    print('Extracting data from {:s}'.format( dbs[0]))
    if db != 'LS3D' :
        Ptr, Gtr = getdata(dbloader, BOT, net)
    else:
        Ptr, Gtr = get_truncated_data(dbloader, BOT, net, max = 400)


    database2 = Full_DB(path=data_path,size=128,flip=False,angle=0.0,tight=tight or 64, db=dbs[1], affine=True)
    dbloader2 = DataLoader(database2, batch_size=30, shuffle=False, num_workers=num_workers, pin_memory=False)
    # extract data        
    print('Extracting data from {:s}'.format( dbs[1]))
    if db != 'LS3D':
        Ptr2, Gtr2 = getdata(dbloader2, BOT, net)
    else:
        Ptr2, Gtr2 = get_truncated_data(dbloader2, BOT, net, max = 30)

    data = {}
    for db in dbs:
        if db not in data.keys():
            data[db] = {}    
    data[dbs[0]][str(epoch)] = {'Ptr': Ptr,  'Gtr': Gtr }
    data[dbs[1]][str(epoch)] = {'Ptr': Ptr2, 'Gtr': Gtr2}        
    
    return data



if __name__ == '__main__':
    main()
