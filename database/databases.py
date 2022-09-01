import inspect, torch, cv2, os, glob, numpy as np, scipy.io as sio, torchfile
from torch.utils.data import Dataset
from utils_fan.utils import process_image, crop, affine_trans

class Full_DB(Dataset):

    def __init__(self, path=None, sigma=1, size=128, step=15, flip=False, angle=0, tight=16, nimages=2, affine=False, db='CelebA'):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        del args[0] 
        for k in args:
            setattr(self,k,values[k])
        preparedb(self,db)
        self.db = db

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        image, points = self.collect(self,idx)
        nimg = len(image)
        if not self.affine:
            sample = dict.fromkeys(['Im', 'ImP'], None)
            flip = np.random.rand(1)[0] > 0.5 if self.flip else False # flip both or none
            for j in range(self.nimages):
                out = process_image(image=image[j%nimg],points=points[j%nimg],angle=(j+1)*self.angle, flip=flip, size=self.size, tight=self.tight)
                if j == 1:
                   sample['Im'] = out['image']
                else:
                   sample['ImP'] = out['image']
        else: 
            tmp_angle = np.clip(np.random.randn(1) * self.angle, -40.0, 40.0) if self.angle > 0 else self.angle
            image,points,_ = affine_trans(image[0],points[0], tmp_angle)
            image, points = crop(image, points, 128, tight=self.tight)
            tmp_angle = np.random.randn(1) * self.angle
            imrot,ptsrot,M = affine_trans(image,points, size=128, angle=tmp_angle)
            image = image/255.0
            image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
            image = image.type_as(torch.FloatTensor())
            imrot = torch.from_numpy(imrot/255.0).permute(2,0,1).type_as(torch.FloatTensor())
            sample = {'Im': image, 'ImP': imrot, 'M' : torch.from_numpy(M).type_as(torch.FloatTensor()) , 'pts': points}
        return sample


def preparedb(self, db):
 
    if db == 'CelebA':
        def init(self):
            txt_file = open(self.path + 'list_landmarks_align_celeba.txt','r')
            lines = txt_file.readlines()[2::]
            names = [l.split()[0] for l in lines]
            coords = [l.split()[1::] for l in lines]
            data = dict(zip(names,coords))
            mafl = [l.replace('\n','') for l in open(os.path.join('database','MAFL_test.txt'),'r').readlines()] # remove MAFL test from training
            files = list(set(names) - set(mafl))
            keys = ['files', 'data']
            for k in keys:
                setattr(self, k, eval(k))
            setattr(self,'len', len(files))
        def collect(self,idx):
            path_to_img = self.path + self.files[idx]
            image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
            points = self.data[self.files[idx]]
            points = np.float32(self.data[self.files[idx]]).reshape(-1,2)
            return [image], [points]
        init(self)
        setattr(self,'collect',collect)

    if 'AFLW' in db:
        def init(self):
            partition = db.split('-')[1]
            mymat = sio.loadmat(os.path.join(self.path, f'aflw_{partition}_keypoints.mat'))
            with open(os.path.join(self.path, f'aflw_{partition}_images.txt'),'r') as f:
                alllines = f.readlines()
            for i in range(len(alllines)):
                alllines[i] = alllines[i].replace('\n','')
            files = alllines
            pts = mymat['gt'][:,:,[1,0]]
            keys = ['files','pts']
            for k in keys:
                setattr(self, k, eval(k)) 
            setattr(self,'len', len(files) )  
        def collect(self,idx):
            path_to_img = os.path.join( self.path, 'output', self.files[idx] )
            image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
            points = self.pts[idx]
            return [image], [points]
        init(self) 
        setattr(self,'collect', collect) 


    if 'MAFL' in db:
        def init(self):
            partition = db.split('-')[1]
            txt_file = open(self.path +'list_landmarks_align_celeba.txt', 'r')
            lines = txt_file.readlines()[2::]
            names = [l.split()[0] for l in lines]
            coords = [l.split()[1::] for l in lines]
            data = dict(zip(names,coords))
            files = [l.replace('\n','') for l in open(f'database/MAFL_{partition}.txt','r').readlines()] 
            notfound = ['031524.jpg', '179577.jpg', '139803.jpg'] if partition == 'train' else []
            for f in notfound:
                idx = files.index(f)
                del files[idx]
            keys = ['files', 'data']
            for k in keys:
                setattr(self, k, eval(k))
            setattr(self,'len', len(files))        
        def collect(self,idx):
            path_to_img = self.path + self.files[idx]
            image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
            points = self.data[self.files[idx]]
            points = np.float32(self.data[self.files[idx]]).reshape(-1,2)
            return [image], [points]
        init(self)
        setattr(self,'collect', collect)

    if 'LS3D' in db: 
        def init(self):
            self.partition = db.split('-')[1]
            if self.partition == 'test' or self.partition == 'Test': 
                self.path = self.path.replace("300W_LP", "LS3D")
                files =  glob.glob(self.path + '/*.jpg', recursive=True)
                data = [w.replace('.jpg', '.t7') for w in files]
            else:
                f_all = glob.glob(self.path + '/**/*.jpg', recursive=True)
                files = [x for x in f_all if '_Flip' not in x]
                data = []
                for w in files:
                    temp = w.split('/')
                    temp.insert(-2,'landmarks')
                    temp = os.path.join(*temp).replace('.jpg', '_pts.mat')
                    data.append(temp)
            keys = ['files','data']
            for k in keys:
                setattr(self, k, eval(k)) 
            setattr(self,'len', len(self.files))
        def collect(self,idx):
            path_to_img = self.files[idx]
            image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
            path_to_pts = self.data[idx]
            if self.partition == 'test' or self.partition == 'Test':
                points = torchfile.load(path_to_pts)
            else:
                points =sio.loadmat(path_to_pts)['pts_3d']
            return [image], [points]
        init(self) 
        setattr(self,'collect', collect) 

    if 'Cats' in db: 
        def init(self):
            self.partition = db.split('-')[1]
            all_files = glob.glob(self.path + '/**/*.jpg', recursive=True)
            if self.partition == 'test' or self.partition == 'Test':
                files = all_files[int(len(all_files)*0.75):]
                data = [x.replace('.jpg','.jpg.cat') for x in files]
            else:
                files = all_files[:int(len(all_files)*0.75)]
                data = [x.replace('.jpg','.jpg.cat') for x in files]
            keys = ['files','data']
            for k in keys:
                setattr(self, k, eval(k)) 
            setattr(self,'len', len(self.files))
        def collect(self,idx):
            path_to_img = self.files[idx]
            image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
            path_to_pts = self.data[idx]
            with open(path_to_pts,'r') as f:
                points = f.read().split(' ')
            points = [int(x) for x in points if x!='']
            points = np.int32(points[1:]).reshape(-1,2)
            return [image], [points]
        init(self) 
        setattr(self,'collect', collect) 


    if 'Shoes' in db: 
        def init(self):
            self.partition = db.split('-')[1]
            all_files = glob.glob(self.path + '/**/**/**/*.jpg', recursive=True)
            all_files = [x for x in all_files if '.jpg' in x]
            if self.partition == 'test' or self.partition == 'Test':
                files = all_files[int(len(all_files)*0.75):]
            else:
                files = all_files[:int(len(all_files)*0.75)]
            keys = ['files']
            for k in keys:
                setattr(self, k, eval(k)) 
            setattr(self,'len', len(self.files))
        def collect(self,idx):
            path_to_img = self.files[idx]
            image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
            points = np.int32([0,0,image.shape[0],image.shape[1]]).reshape(-1,2) 
            return [image], [points]
        init(self) 
        setattr(self,'collect', collect) 

   