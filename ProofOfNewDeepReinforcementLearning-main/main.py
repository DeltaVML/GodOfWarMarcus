


import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


from tensorflow.keras.layers import Dense,Conv2D,Flatten,Activation,Input,MaxPool2D

from model_enviorn import TestModelEnviorment


from custom_callback import custom_callback


INPUT_SHAPE = [int(250//2), int(160//2), 3]




# if (tf.keras.backend == 'tensorflow'):
tf.keras.backend.clear_session()

# np.random.seed(42)
# tf.random.set_seed(42)


class worker_main():

    def __init__(self):
        pass 
        
    def define_model(self,action_size,gpu):



        x_input = Input(shape=INPUT_SHAPE)
        x = Conv2D(32,activation='relu',kernel_size=[3,3],strides=[1,1],input_shape=INPUT_SHAPE,padding='SAME')(x_input)
        x = Conv2D(64,activation='relu',kernel_size=[3,3],strides=[1,1],input_shape=INPUT_SHAPE,padding='SAME')(x)
        x = MaxPool2D([2,2])(x)
        x = MaxPool2D([2,2])(x)
        x = Conv2D(16,activation='relu',kernel_size=[3,3],strides=[1,1],input_shape=INPUT_SHAPE,padding='SAME')(x)


        x = Flatten()(x)

        x = Dense(64,activation='relu')(x)
        v_s = Dense(1)(x)
        x = Dense(action_size,activation='linear')(x)


        
        mx_steps = 10000


     
        model = TestModelEnviorment(n_step=32,env="ALE/AirRaid-v5",action_size=action_size,epochs=1000,reward_function=lambda data: (data) / mx_steps,max_steps=mx_steps,obs_shape=[250,160,3],obs_resized_shape=INPUT_SHAPE[:-1],reward_total_cond_not=False,discount_factor=0.99,skip_n=4,gpu=gpu,inputs=[x_input],outputs=[x,v_s])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005),loss='mse')


        # gpu: 0 .99
        # gpu: 0.43
        return model

    def run_test(self,gpu):
        


        model = self.define_model(6,gpu) 
        
        x_n = tf.constant(np.arange(int(1e5)))

        # tf.data.Dataset.from_tensor_slices((),())
        train_data = tf.data.Dataset.from_tensor_slices(x_n).batch(1).prefetch(2)
        # test_data = tf.data.Dataset.from_tensor_slices(x_n,x_n).batch(1).prefetch(2)        


        model.fit(train_data,epochs=1,batch_size=1,callbacks=[custom_callback()])

        # import pdb; pdb.set_trace()

        import pdb; pdb.set_trace()



if (__name__ == '__main__'):


    import sys
    gpu = int(sys.argv[1].split('=')[1])

    # no eps running test.
    with tf.device(f'GPU:{gpu}'):



        worker_main().run_test(gpu)

    # eps expodntial running test.
    # with tf.device('GPU:1'):
    #     worker_main().run_test()

    # Eplison not running test.
    # with tf.device('GPU:1'):
    #     worker_main().run_test()