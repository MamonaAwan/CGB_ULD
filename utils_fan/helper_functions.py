
import glob, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from utils_gcn.misc import l2norm, write_feat, read_probs
from shutil import rmtree

def save_npz(ofn, labels):
    if os.path.exists(ofn):
        print('Overwritting Labels file!')
    np.savez_compressed(ofn, data=labels)

def write_features(ofn, features):
    print('save features to', ofn)
    with open(ofn, 'ab') as ofn:
        features.tofile(ofn)


def  store_landmark_features(lo_hypcol_rp, d=256):
    '''
        Function for storing landmark features, to be loaded into gcn afterwards.
        input: 
            (lo_hypcol_rp) = [batch_size* iterations, num_of_landmarks, 256]
        output:
            stores features in required form at fixed paths.
    
    '''
    files = glob.glob('data/features/*')
    for f in files:
        os.remove(f)
    if os.path.exists('data/knns/test'):
        rmtree('data/knns/test', ignore_errors= False)

    lo_file_path = os.path.join('data','features','{}.bin'.format('hypercol_rp'))
    
    # dump features in bin files
    write_feat(lo_file_path, lo_hypcol_rp.view(-1, d).data.cpu().numpy())
    
    
    # mine K-NN (affinity graph)
    lo_rp_feat = l2norm(lo_hypcol_rp.view(-1, d).data.cpu().numpy())

    print('lo_rp_feat_this = ', lo_rp_feat.shape)
    print('-------- Representation Features Stored in {} !----------'.format(lo_file_path))


def load_cluster_centroids(predictions_dir):
    '''
        Function to load and calculate centroids from gcn clusters
        output:
            label_centroids= [num_of_clusters, centroids]
        (where label is the index of dim 0 and centroid is whole dim 1) 

    '''
    labels_path = predictions_dir
    features_path = 'data/features/hypercol_rp.bin'

    label_np = np.loadtxt(labels_path, dtype ='int')    
    features_np = read_probs(features_path, -1, 256)
 
    assert label_np.shape[0] == features_np.shape[0], "Landmark representations & Labels number mismatch!"
    
    lbl_ = np.zeros((max(label_np)+1, features_np.shape[1]), dtype = np.float32)
    cnts = np.zeros((max(label_np)+1, 1), dtype = int)

    for (lbl, ft) in zip(label_np, features_np):
        lbl_[lbl] = np.add(lbl_[lbl], ft)
        cnts[lbl] += 1
     
    label_centroids = torch.from_numpy(l2norm(np.divide(lbl_, cnts)))
    fts_per_clusters = torch.from_numpy(cnts)

    print('Num of Representations used = {}'.format(label_np.shape[0])) 
    print('Centroids = {},  Centroid Labels = {} '.format(label_centroids.shape, fts_per_clusters.shape))
        
    return label_centroids, fts_per_clusters



def assign_pseudo_label(batch_landmarks, label_centroids, fts_per_clusters, update_centroids = False):
    # label is index of dim=0 and centroid is whole dim=1 in label_centroids
    '''
        Function for assignment of cluster/pseduo label based on similarity.
        input 
            batch_landmarks = [batch_size, num_of_landmarks, 256]
            label_centroids = [labels, 256]
            fts_per_clusters = [labels, count_of_fts_in_cluster]
            update_centroids = bool (if true updated centroids and fts_per_cluster are returned)
        output:
            csim             = [batch_size , num_of_landmarks]       (2-D configuration)
    '''
    d = 256 # dimension set as required by gcnv
    b_size, num_of_landmarks = batch_landmarks.shape[0], batch_landmarks.shape[1]
    batch_landmarks = batch_landmarks.view(-1, d)
    b_lm2 = F.normalize(batch_landmarks, dim = 1, p =2).to(torch.float64)
    label_centroids = label_centroids.to('cuda')
    
    sim_mat = torch.matmul(b_lm2, label_centroids.T) 
    max_csim,labels = torch.max(sim_mat, dim=1)
    max_csim = max_csim.view(b_size, num_of_landmarks)
    

    return max_csim
 

class rep_extractor():
    def __init__(self, dim = 768):
        super(rep_extractor, self).__init__()
        self.conv_1 = nn.Conv2d(dim, 256, 1, stride = 1, padding = 0).cuda() # using 256 for dim of representations
        self.ada_pool = nn.AdaptiveMaxPool2d(1).cuda()
        

    def get_landmark_repr(self, hm, fts, reduce_fts_dim = False):
        '''
            Function to get Landmark Representations from Heatmaps and Repr Col
            input: 
                heatmaps (hm) =     [batch_size, hmaps, h, w]
                repr (fts) = [batch_size, D, h, w]
                reduce_fts (bool) = reduce the D dimension of repr if True
            output:
                landmark representations = [batch_size, hmaps, repr]

        '''
        # split heatmaps (batch_size, hmaps, h, w) into list  [(batch_size, 1, h, 2), ... ]
        hm_splited = torch.split(hm, 1, 1) 

        # function to get 1 landmark representation from 1 heatmap
        def get_one_repr(hm_one):
            assert fts.shape[0] == hm_one.shape[0], "Invalid Features with Heatmaps numbers"
            if reduce_fts_dim:
                fts_rdim = self.conv_1(fts)
                
            fts_ = torch.matmul(fts_rdim, hm_one) if reduce_fts_dim else torch.matmul(fts, hm_one)
            fts_ = self.ada_pool(fts_) 
            
            landmark_repr_one = fts_.view(fts_.shape[0], -1).unsqueeze(1)
            return landmark_repr_one

        # concatenate all Landmark representations
        landmark_repr = torch.cat([get_one_repr(hm_one) for hm_one in hm_splited], 1)
        # print("landmark_repr shape = ", landmark_repr.shape)
        return landmark_repr


