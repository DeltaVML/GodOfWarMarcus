

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 




from tensorflow.keras.layers import Dense,Reshape,Activation,Input,Flatten
 

from custom_layers import PosEmb,NuronExchange,Expand,Slicing

from model_enviorn_foud import model_enviorn_foud

from enc_env import enc_env


from custom_callback import custom_callback



class main:
    
    
    def __init__(self):
        
        
        self.input_shape = [128,1]
        self.action_shape = [32,1]
        self.vocab_size = 66
        self.end_vocab_size = 97

        self.batch_size = 1024
        
        
    
    
    def define_model(self):
    
        x_input = Input(shape=self.input_shape[:1])
        
        x = PosEmb(2048,self.vocab_size)(x_input)



        x = Dense(32,activation='relu',input_shape=self.input_shape[:-1] + [2048])(x)
        x = Dense(64,activation='relu',input_shape=self.input_shape[:-1] + [32])(x)
        x = Dense(256,activation='relu',input_shape=self.input_shape[:-1] + [64])(x)
        x = Dense(128,activation='relu',input_shape=self.input_shape[:-1] + [256])(x)

        x = Slicing(32)(x)
        v_s = Dense(1,input_shape=self.input_shape[:-1]+[128])(x)
        x = Dense(self.end_vocab_size,input_shape=self.input_shape[:-1]+[128])(x)



        model = model_enviorn_foud(n_step=self.batch_size,action_size=self.end_vocab_size,reward_function=lambda data: data,max_steps=2048,input_shape=self.input_shape,discount_factor=0.98,env=enc_env(self.input_shape[0],self.action_shape[0]),apply_weights=False,seq_len=self.action_shape[0],inputs=[x_input],outputs=[x,v_s])
        
        model.compile(loss=['mse'],optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005),run_eagerly=False)


        return model
        
        
    
    def train_loop(self):
        
        
        
        model = self.define_model()
        
        data_numpy_fake = np.arange(int(1e5)).astype(np.float32)
        data = tf.data.Dataset.from_tensor_slices((data_numpy_fake,data_numpy_fake)).batch(1).prefetch(2)
        
        model.fit(data,epochs=1,batch_size=1)
        model.save_weights('main.h5')
        import pdb; pdb.set_trace()
        
        

if (__name__ == '__main__'):
    
    with tf.device('GPU:0'):
        main().train_loop()