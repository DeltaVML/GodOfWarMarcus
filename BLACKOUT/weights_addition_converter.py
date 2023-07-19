






import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2
import tensorflow as tf 




class weights_addition_converter:
    

    def __init__(self,layer_end=2):
        

        
        self.layer_end = layer_end
        
        
        

    
    def adaptive_gen_actions(self,x):
        
        maxY = np.max(np.abs(x))
        
        return np.linspace(-1,1,100) * (maxY*1.3)

         
    def test_function(self,weights):
        
        self.test_model_new.set_weights(weights)
        
        return self.test_model_new.test_step_fast(self.test_agent_dataset).numpy().mean(axis=0) 
        
        
        
    def model_weights_fomular(self,og_weights_indexed,new_values,expand=False):
        
        
        
        
        
        applied = [ np.expand_dims(og_weights_indexed,axis=0)[:,data] if expand else np.expand_dims(og_weights_indexed[:,data],axis=-1) for data in np.arange(og_weights_indexed.shape[-1])]
        applied.append(new_values)
        
        
        # if (og_weights_indexed.shape[0] == 64):
        #     import pdb ; pdb.set_trace()
        
        return tf.concat(applied,axis=-1).numpy()
        



    # actions n-layers-adaptive,nurons,inner_nuron_layer_backprop_applied. 

    # Output: -1,0 reward space. float64
    def step(self,actions,weights):
        
        x = weights
        
        for index in (np.arange(self.layer_end)+1)*-1:
            
            indexed_weights = x[index].copy()
            x[index] = self.model_weights_fomular(indexed_weights,self.adaptive_gen_actions(indexed_weights)[actions[index]],expand=index ==-1 )
            
        

        
            
        return x

        
        


if (__name__ == '__main__'):
    
    rewards = WeightsAgentEnv(layer_end=2,trainCond=True).step([np.zeros(shape=[64,1]).astype(np.int32),np.ones(shape=[1]).astype(np.int32)],)
    
    import pdb ; pdb.set_trace()
    
    
    