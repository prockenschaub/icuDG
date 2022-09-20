
import numpy as np

class SyntheticDataset():
    def __init__(self, num_envs, num_smpls, seed=None):        
        rand_state = np.random.RandomState(seed)
        self.ds = [SyntheticDataset.simulate_env(num_smpls, rand_state) for _ in range(num_envs)]
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def simulate_env(num_smpls, rand_state=None):
        # https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5
        theta = np.sqrt(rand_state.rand(num_smpls))*2*np.pi 

        r_a = 2*theta + np.pi
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + 4 * rand_state.randn(num_smpls,2)

        r_b = -2*theta - np.pi
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        x_b = data_b + 4 * rand_state.randn(num_smpls,2)

        res_a = np.append(np.zeros((num_smpls,1)), x_a, axis=1)
        res_b = np.append(np.ones((num_smpls,1)), x_b, axis=1)

        res = np.append(res_a, res_b, axis=0)
        rand_state.shuffle(res)
        
        return res.astype(np.float32)
