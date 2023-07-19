

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 


import os 


class enc_env:
    
    
    def __init__(self,seq_len=128,action_seq_len=32):
        
              
        self.max_steps_log = 300
        self.seq_len = seq_len
        self.action_seq_len = action_seq_len
        self.main_data_dir = '/home/server28102508/HacksData'  
        self.dirs_main  = np.array(os.listdir(self.main_data_dir))


        self.vocab = {}
        self.value_vocab = {}
        

        index = 0
        for data in np.array([ chr(data) for data in np.arange(256)]):   
            self.vocab[data] = index 
            index += 1
                
                
                
        # self.str_data = np.array([ chr(data) for data in np.arange(1000)])[32:127]
        
        index = 0
        for data in np.array([ chr(data) for data in np.arange(1000)])[32:127]:
            self.value_vocab[data] = index
            index += 1
        
        self.bundel()
        self.enc_file_values = None 
        self.file_ops_init()
    

    def pad_sequence(self,data):

        if (data.shape[0] != self.seq_len):

            data = np.append(data,np.zeros(shape=[ self.seq_len - data.size]))



        return data
    
    def file_ops_init(self):
        
        self.enc_files_args = np.array([])
        
        for data in np.array(os.listdir(self.main_data_dir)):
            

            for data_number in np.array(os.listdir(self.main_data_dir+'/'+data)):
                self.enc_files_args = np.append(self.enc_files_args,data+'/'+data_number)
    

    def bundel(self):
        
        
        self.budel_data_nps = np.array([])
        
        for data in self.dirs_main:
            dir_defined = self.main_data_dir+'/'+data
            dir_x = np.array(os.listdir(dir_defined))
            
            for data_main_dir in dir_x:
                
                self.budel_data_nps = np.append(self.budel_data_nps,dir_defined+'/'+data_main_dir)

        self.budel_index = 0
        self.enc_smaple_index = 0
        self.enc_provo = 0
        np.random.shuffle(self.budel_data_nps)
        
        self.enc_samples = self.budel_data_nps[self.enc_smaple_index]
    
    
    def vocab_numpy_from_str(self,data_main):
        return np.array([self.vocab[data] for data in data_main ])
    
    
            
    def get_index_numpy(self):
        
        self.enc_provo += 1
        
        
        try:
            if (self.enc_provo == self.enc_provo_max):
                

                self.enc_smaple_index += 1

                
                if (self.enc_smaple_index == self.enc_samples.size):
                    self.enc_smaple_index += 1

                    self.budel_index += 1
                    
                    if (self.budel_index == self.budel_data_nps.size):
                        
                        self.budel_index = 0
                            
                        np.random.shuffle(self.budel_data_nps)
                        
                    self.enc_samples = np.load(self.budel_data_nps[self.budel_index])
                    
                        
        except:
            self.enc_samples = np.load(self.budel_data_nps[self.budel_index]).reshape(-1,3)


        sample = self.enc_samples[self.enc_smaple_index]
        
        key_ytrue = self.vocab_numpy_from_str(sample[1])



        return (key_ytrue,sample[1],np.array([ self.value_vocab[data] for data in sample[2]]))
        

    def step(self,action):
        
        key_ytrue,sample,sample_2 = self.get_index_numpy()


        reward = (self.ytrue.astype(np.int32) == action.argmax(axis=-1)).astype(np.float32)
        
        # reward = np.append(reward - np.ones(shape=[self.ytrue.shape[0]]),np.zeros(shape=[action.shape[0] - self.ytrue.shape[0] ]))
        
        

        reward = reward -1

        self.ytrue = key_ytrue

        
        self.step_index_log += 1
        
        
        done = self.max_steps_log == self.step_index_log



        if (done):
            self.step_index_log = 0
            

        return (self.pad_sequence(sample_2),reward,done,{})    
    
    def reset(self):
        
        
        self.step_index_log = 0

        
        self.ytrue,sample,obs = self.get_index_numpy()

        return self.pad_sequence(obs)
    
    
if (__name__ == "__main__"):
    
    enc_env()