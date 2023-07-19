








# from DeepEnv import DeepEnv


import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from custom_functions import custom_exp_lower_base
from proccess_functions import proccess_functions
    



@tf.function()
def base_reward_tf(data):
    return data 


class model_enviorn_foud(proccess_functions):
    
    
    
    def __init__(self,n_step,action_size,reward_function,max_steps,input_shape,discount_factor
                 ,env,apply_weights=False,n_factor=16,seq_len=32,*args, **kwargs):

            
        
        super().__init__(*args, **kwargs)
        
   
        self.env = env
        
        self.loss_current = None 

        self.apply_weights = apply_weights
        self.n_step = n_step
        self.action_size = action_size
        self.step = 0
        self.max_steps = max_steps
        self.eps = 1.0
        self.obs_shape = input_shape[:1]

        self.reward_step = np.array([])
        self.reward_five_rul = 0
        


        self.reward_function = reward_function
        self.discount_factor = discount_factor
        
        self.initlized_env = False 
        

        self.seq_len = seq_len
        self.n_factor = n_factor

        # self.action_nested_shapes = [-1,self.seq_len] + [self.n_factor,self.action_size]
        # self.action_nested_shapes_flattend = [-1,self.seq_len] + [1,1]
        # self.action_nested_shapes_flattend_vs_x = [-1,self.seq_len] + [16,1]

        self.action_nested_shapes = [-1,self.seq_len] + [self.action_size]
        self.action_nested_shapes_flattend = [-1,self.seq_len] + [1]
        self.action_nested_shapes_flattend_vs_x = [-1,self.seq_len] + [1]  

        self.epoch = -1
        self.done = True 
        self.obs = None 





        

    def reset(self):
        



        if (self.done):
            self.epoch += 1
            new_obs = self.env.reset()
            self.done = False
            self.step = 0
            self.reward_total_actutal = np.zeros(shape=self.seq_len)
            self.obs = new_obs
        else:
            new_obs = self.obs
    



        return np.array(new_obs).astype(np.float32) 


    def reset_tf(self):
        
        obs = tf.numpy_function(self.reset,[],[tf.float32])

        return tf.reshape(obs,self.obs_shape)
    
    def get_first_node_obs_n(self):
        return self.recusive_function.first_node_obs_n().astype(np.float32)

        
     
    def get_first_node_obs_n_tf(self):
        
        tf.numpy_function(self.get_first_node_obs_n,[],[tf.float32])
    
    
    def change_n_step_dpf(self,n_step_dpf):
        
        self.n_step = n_step_dpf

    def change_n_step_dpf_tf(self,n_step_dpf):
        
        return tf.numpy_function(self.change_n_step_dpf,[n_step_dpf],[])
    
    



    def step_function(self,action : np.ndarray):
        obs,reward,done,info = self.env.step(action)
        
        self.step += 1
        # reward = reward - (1/self.obs_shape_prod*16*12)
        self.obs = obs



        
        self.reward_total_actutal += (reward.reshape(self.seq_len)+1)
        # reward = reward - (0.1/(self.max_steps*30))
        self.reward_total = reward 
        # self.reward_total += np.prod()


        if (self.step == self.max_steps or done):
            self.done = True 
            done = True 




        return (np.array(obs).astype(np.float32),np.array((self.reward_total)*(self.discount_factor ** self.epoch) ).astype(np.float32),np.array(done).astype(dtype=np.int32))


    def step_tf(self,action):

        obs,reward,done = tf.numpy_function(self.step_function,[action],[tf.float32,tf.float32,tf.int32])




        return (tf.reshape(obs,self.obs_shape),tf.reshape(reward,[-1,1]),tf.reshape(done,[1])[0])

    
    def train_episode(self,obs):
        

    
        
        rewards = tf.TensorArray(size=0,dynamic_size=True,dtype=tf.float32)
        actions = tf.TensorArray(size=0,dynamic_size=True,dtype=tf.float32)
        dones = tf.TensorArray(size=0,dtype=tf.int32,dynamic_size=True)
        v_sX = tf.TensorArray(size=0,dtype=tf.float32,dynamic_size=True)
    
        v_sX_new = tf.TensorArray(size=0,dtype=tf.float32,dynamic_size=True)
        actions_new = tf.TensorArray(size=0,dtype=tf.float32,dynamic_size=True)
        

        reward_total = tf.cast(0.0,dtype=tf.float32)



        for data in tf.range(self.n_step):
            



            action,v_s = self(tf.expand_dims(obs,axis=0))
            


            # action = tf.squeeze(action,axis=0)
            
            obs,reward,done = self.step_tf(action)

            # obs,reward,done = self.recusive_function(self.process_function_for_obs(action))
            # values =  model_init.train_step(action)

            reward_total += tf.reduce_mean(reward) 
            
            v_sX = v_sX.write(data,v_s)
            rewards = rewards.write(data,reward)

            actions = actions.write(data,action)
            dones = dones.write(data,done)

            action,v_s = self(tf.expand_dims(obs,axis=0))
                
            v_sX_new = v_sX_new.write(data,v_s)
            actions_new = actions_new.write(data,action)
                        
            
            if tf.cast(done,tf.bool):
                break
            
        
 
    
        
        return (actions.stack(),rewards.stack(),dones.stack(),v_sX.stack(),v_sX_new.stack(),actions_new.stack(),reward_total)
    
     
    

     
    @tf.function()
    def train_step(self,data):
        

        obs = self.reset_tf()
        
        
        
        with tf.GradientTape() as tape:
            
            actions,rewards,dones,v_sX,v_sX_new,actions_new,reward_total = self.train_episode(obs) 

            # self.update_to_current_batch_lvl_tf(dones)

            loss_x = self.compute_loss(actions,rewards,v_sX,v_sX_new,actions_new)



        
        grad = tape.gradient(loss_x, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        
        self.did_nan_tf(loss_x)
        
        return {'loss':tf.reduce_mean(loss_x),'reward_total':tf.reduce_mean(reward_total)}
        
    
    
    
    def check_is_nan_cond(self,value,extra,value_real,extra_real,output):
        pass 

        # if (value > 0 or extra > 0):



    def check_is_nan_cond_tf(self,value,extra,output):
        tf.numpy_function(self.check_is_nan_cond,[tf.reduce_sum(tf.cast(tf.math.is_inf(value),dtype=tf.float32)),tf.reduce_sum(tf.cast(tf.math.is_inf(extra),dtype=tf.float32)),value,extra,output],[])
    
    
    def update_obs(self,obs):
        
        self.obs = obs 
    
    def update_obs_tf(self,obs):
        tf.numpy_function(self.update_obs,[obs],[])
    

        
    
    def did_nan(self,loss_x: np.ndarray):

        self.did_nan_cond = loss_x

        if (self.did_nan_cond):
            import pdb; pdb.set_trace()

    
    def did_nan_tf(self,loss_x):
        return tf.numpy_function(self.did_nan,[tf.math.is_nan(tf.reduce_mean(loss_x))],[])
    
    

        
        
    #  RESET_ENV -> C_net x_y -> Recusive x_y -> A_net x_y -> Env x_y
    
    
        
    def update_main_env_real_after_step(self):
        
        return self.env.recusive_function.env         

    def update_main_env_real_after_step_tf(self):
        
        tf.numpy(self.update_main_env_real_after_step,[],[])


    def end_iteration(self,x1,x2,x3,x4):
        print(self.action_nested_shapes_flattend_vs_x)
        print(x1.shape,x2.shape,x3.shape,x4.shape)


    def end_iteration_tf(self,x1,x2,x3,x4):
        tf.numpy_function(self.end_iteration,[x1,x2,x3,x4],[])




    def compute_loss(self,actions,rewards,v_sX,v_sX_new,actions_new):
        
        # Both be equal to (None)
        



        actions = tf.reshape(tf.cast(actions,dtype=tf.float32),self.action_nested_shapes)
        rewards = tf.reshape(tf.cast(rewards,dtype=tf.float32),self.action_nested_shapes_flattend)


        v_sX = tf.reshape(tf.cast(v_sX,dtype=tf.float32),self.action_nested_shapes_flattend_vs_x)
        v_sX_new = tf.reshape(tf.cast(v_sX_new,dtype=tf.float32),self.action_nested_shapes_flattend_vs_x)
        actions_new = tf.cast(tf.reshape(actions_new,self.action_nested_shapes),dtype=tf.float32)
        

        action_loss = tf.math.lgamma(tf.where(tf.math.is_inf(tf.divide(1.0,actions_new)),actions_new+1e6,actions_new)) 
        self.check_is_nan_cond_tf(action_loss,actions_new,'action_loss')


        exp_value = custom_exp_lower_base(actions)
        self.check_is_nan_cond_tf(exp_value,actions,'exp_value')



        # GPU-0
        
        
        # self.end_iteration_tf(actions,rewards,v_sX,v_sX_new)
        
        # if (data == 0):
        #     rewards = tf.expand_dims(tf.reshape(tf.repeat(rewards,axis=0,repeats=tf.cast(tf.reduce_prod(tf.shape(action_loss)) / (128*10*10*12),dtype=tf.int32)),tf.shape(action_loss)[:-1]),axis=-1)
        # else:
        #     rewards = tf.reshape(rewards,[128,10,10])
            
        #     rewards = tf.expand_dims(tf.repeat(tf.expand_dims(tf.reduce_sum(tf.reduce_sum(rewards,axis=-1),axis=-1) / 100,axis=-1),axis=-1,repeats=tf.shape(action_loss)[1]),axis=-1)

        # if (data == 1):
        #     import pdb; pdb.set_trace()

        loss_x = (tf.cast(action_loss * v_sX_new,dtype=tf.float32) * ((rewards * exp_value )*v_sX)/10)
        # x = tf.zeros(shape=[23,2332])

        loss_x = tf.math.lgamma(tf.where(tf.math.is_inf(tf.divide(1.0,loss_x)),loss_x+1e6,loss_x)) 

        return loss_x