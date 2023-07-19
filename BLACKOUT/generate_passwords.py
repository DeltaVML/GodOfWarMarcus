


# import tensorflow as tf 
import numpy as np 
import tensorflow as tf 
# import pandas as pd 


class generate_passwords:
    
    
    def __init__(self,path='path.txt'):
        
        
        self.word_list = path
        
        self.length_of_generation = 12
        self.index = 0
        self.combines = np.array([])
        self.values = np.array([])
        
        self.combined_combs = np.array([])

        for data in np.array([['0','9'],['a','z'],['A','Z']]):
            self.combined_combs = np.append(self.combined_combs,np.arange(ord(data[0]),ord(data[1])))
        
        self.combined_combs = np.array([ str(chr(data)) for data in self.combined_combs.astype(np.int32)])

        comb = self.added_extetion(self.combined_combs)
        import pdb; pdb.set_trace()


    # @tf.function()
    def added_extetion(self,comb):
        
        stacks = tf.TensorArray(dtype=tf.float32,size=0)
        
        comb_new = comb
        for data in np.arange(30):
            
            comb_insert = comb_new[0]

            comb_new = tf.concat([tf.slice(comb,[data]),comb_insert],axis=0)
            
            
            stacks = stacks.write(data,tf.expand_dims(comb_new,axis=1) + tf.expand_dims(comb_insert,axis=-1))
            


    def length_stretch(self,words):
        
        
        name_bag = np.load('name_bag.npy')
        vocab_bag = np.load('vocab.npy')
        
        
        
        
        
        
        
        
        
    
            
if (__name__ == '__main__'):
    
    generate_passwords().generate_test_bags()
        
        
        
        
        
        
        