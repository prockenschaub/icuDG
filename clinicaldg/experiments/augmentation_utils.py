import numpy as np

def corrupt(col, p):
    return np.logical_xor(col, np.random.binomial(n = 1, p = p, size = len(col)))  

def compute_subsample_probability(subset_df, desired_prob, target_name = 'target'):
    pos = subset_df[target_name].sum()
    cur_pos = pos/len(subset_df)

    if cur_pos >= desired_prob:            
        p = 1 - (1 - cur_pos)/cur_pos * (desired_prob)/(1-desired_prob)
        return (1,  p) # subsample positive samples
    else:
        p =  1 - cur_pos/(1-cur_pos) * (1-desired_prob)/(desired_prob)
        return (0,  p) # subsample negative samples
    
def aug_f(grp, target, brackets):    
    a,b = brackets[grp]
    if target == a:
        return b
    else:
        return 0    