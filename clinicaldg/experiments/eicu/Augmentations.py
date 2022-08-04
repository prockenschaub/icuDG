import numpy as np      
from . import experiments
from ..augmentation_utils import (
    corrupt, 
    compute_subsample_probability, 
    aug_f      
)
                
class AddCorrelatedFeature():
    def __init__(self, train_corrupt_dist, train_corrupt_mean, val_corrupt, test_corrupt, feat_name):
        self.feat_name = feat_name
        
        self.corrupts = {experiments.eICU.TRAIN_ENVS[0]: train_corrupt_mean - train_corrupt_dist, 
                    experiments.eICU.TRAIN_ENVS[1]:train_corrupt_mean,
                      experiments.eICU.TRAIN_ENVS[2]: train_corrupt_mean + train_corrupt_dist,
                      experiments.eICU.VAL_ENV: val_corrupt,
                      experiments.eICU.TEST_ENV: test_corrupt}   

        print('CorrLabel parameters: \n' + str(self.corrupts), flush = True)    
        assert(all([i >=0 for i in self.corrupts.values()]))
    
    def augment(self, reg_mort, reg_pat):
        for env in self.corrupts:
            reg_pat[env][self.feat_name] = corrupt(reg_pat[env]['target'], self.corrupts[env])
        
class Subsample():
    def __init__(self, g1_mean, g2_mean, g1_dist, g2_dist):

        self.means = {
            experiments.eICU.TRAIN_ENVS[0]: (g1_mean + g1_dist, g2_mean - g2_dist),
            experiments.eICU.TRAIN_ENVS[1]: (g1_mean, g2_mean),
            experiments.eICU.TRAIN_ENVS[2]: (g1_mean - g1_dist, g2_mean + g2_dist),
            experiments.eICU.VAL_ENV: (0.3, 0.3),
            experiments.eICU.TEST_ENV: (0.1, 0.5)
        }

        print('Subsampling parameters: \n' + str(self.means), flush = True)
    
    def subsample(self, mort_df, pat_df, g1, g2):     
        pat_df['group_membership'] = pat_df['gender'] == 'Male'
        
        brackets = {}
        brackets[True] = compute_subsample_probability(pat_df[pat_df.group_membership], g1)
        brackets[False] = compute_subsample_probability(pat_df[~pat_df.group_membership], g2)
        
        pat_df['prob'] = pat_df[['group_membership', 'target']].apply(lambda x: aug_f(x['group_membership'], x['target'], brackets), axis = 1)
        pat_df['roll'] = np.random.binomial(1, p =  pat_df['prob'].values, size = len(pat_df))
        
        drop_inds = list(pat_df.index[(pat_df['roll'] == 1)])
        
        pat_df = pat_df[~pat_df.index.isin(drop_inds)]
        
        mort_df = mort_df.loc[~mort_df.index.get_level_values(0).isin(drop_inds)]
        
        return mort_df, pat_df    
                
    def augment(self, reg_mort, reg_pat): 
        for env in self.means:
            assert(0 <= self.means[env][0] <= 1)
            assert(0 <= self.means[env][1] <= 1)
            reg_mort[env], reg_pat[env] = self.subsample(reg_mort[env], reg_pat[env], self.means[env][0], self.means[env][1])
                     
            
class GaussianNoise():        
    def __init__(self, train_corrupt_dist, train_corrupt_mean, val_corrupt, test_corrupt, std, feat_name = 'admissionweight'):
        self.feat_name = feat_name
        self.std = std

        self.corrupts = {experiments.eICU.TRAIN_ENVS[0]: train_corrupt_mean - train_corrupt_dist, 
                    experiments.eICU.TRAIN_ENVS[1]:train_corrupt_mean,
                      experiments.eICU.TRAIN_ENVS[2]: train_corrupt_mean + train_corrupt_dist,
                      experiments.eICU.VAL_ENV: val_corrupt,
                      experiments.eICU.TEST_ENV: test_corrupt}   

        print('CorrNoise parameters: \n' + str(self.corrupts), flush = True)                    

    def add_noise(self, feat_col, pat_df, mean):
        pat_df['signed_target'] = pat_df['target'] *2 - 1
        pat_df['noise'] = np.random.normal(mean, self.std, size = (len(pat_df), )) * pat_df['signed_target']  
                
        feat_col = feat_col.to_frame(0).apply(lambda x: x[0] + pat_df.loc[x.name[0], 'noise'], axis = 1)
        return feat_col
        
    def augment(self, reg_mort, reg_pat):
        feat_name = self.feat_name       
        for env in self.corrupts:
            reg_mort[env][feat_name] = self.add_noise(reg_mort[env][feat_name], reg_pat[env], self.corrupts[env])