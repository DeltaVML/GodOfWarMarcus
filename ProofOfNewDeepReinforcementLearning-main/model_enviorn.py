




# from DeepEnv import DeepEnv


import tensorflow as tf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import gym 
import cv2
from custom_functions import custom_exp_lower_base,uniform_eps_exp_var


# Build Equation. 


# new_line+math_ops+vars in second ouput, math_ops in one area,
# 30 + n_math_ops.



#  define_step_struc_2 (2) output, signular)
# define_step_struct_1 (1) output)


# IF TOP LEVEL OP COMPLTED >= 1 the new_line action is valid.

# (step_maths,new_line,compile_losses/end/train) intial_vars (reward,action_probs) + step_vars Or step_maths_nested... step_vars(action_remaiders) ... 



# step_vars 

    # (defined as vars), 0-30, that is ordered as e.g. [0,1,2,3...] tf.tile to unless 0,

    



# Reward_Clipped.


    

class TestModelEnviorment(tf.keras.models.Model):
    
    
    
    def __init__(self,n_step,env,action_size,epochs,reward_function,max_steps,obs_shape,obs_resized_shape,reward_total_cond_not,discount_factor,gpu,skip_n,*args, **kwargs):
        
        
        super().__init__(*args, **kwargs)
        
        self.env = gym.make(env)

        self.n_step = n_step
        self.action_size = action_size
        self.step = 0
        self.max_steps = max_steps
        self.eps = 1.0
        self.obs_resized_shape = obs_resized_shape
        self.obs_shape = obs_shape
        # self.eps_array = np.linspace(1,0,epochs*max_steps)
        # self.eps_index = 0 
        self.reward_step = np.array([])
        self.reward_track = np.array([])
        self.reward_five_rul = 0
        self.skip_n = skip_n

        self.reward_function = reward_function
        self.discount_factor = discount_factor
        self.reward_total_actual = 0
        self.gpu = gpu

        self.epoch = -1
        self.done = True 
        self.obs = None 

        self.reward_total_cond_not = reward_total_cond_not
        
        
    
    @tf.function()    
    def train_step(self,data):
        
        
        with tf.GradientTape() as tape:
            actions,rewards,dones,v_sX,v_sX_new,actions_new,reward_total = self.train_episode() 

            loss_x = self.compute_loss(actions,rewards,dones,v_sX,v_sX_new,actions_new)

        
        grad = tape.gradient(loss_x, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        
        self.did_nan_tf(loss_x)
        
        return {'loss':tf.reduce_mean(loss_x),'reward_total':tf.reduce_mean(reward_total)}
        
    
    
    
    def check_is_nan_cond(self,value,extra,value_real,extra_real,output):


        if (value > 0 or extra > 0):
            print(value,output)

            import pdb; pdb.set_trace()

        

    def check_is_nan_cond_tf(self,value,extra,output):
        tf.numpy_function(self.check_is_nan_cond,[tf.reduce_sum(tf.cast(tf.math.is_inf(value),dtype=tf.float32)),tf.reduce_sum(tf.cast(tf.math.is_inf(extra),dtype=tf.float32)),value,extra,output],[])
    
    
    def env_step_reset(self):
        
        
        # self.eps = self.eps_array[self.epoch]
        if (self.done):
            try:
                self.reward_track = np.append(self.reward_track,self.reward_total)
            except:
                pass
            self.reward_total = 0
            self.epoch += 1
            self.reward_total_actual = 0
            new_obs = self.env.reset()[0]/255
            self.done = False
            self.step = 0
            self.obs = new_obs
        else:
            new_obs = self.obs
            
        return new_obs.astype(np.float32)
    
    
    def env_step_tf_reset(self):
        new_obs = tf.numpy_function(self.env_step_reset,[],[tf.float32])
        return tf.image.resize(tf.reshape(new_obs,self.obs_shape),self.obs_resized_shape)
    

    def frame_skip(self,action):

        skips = np.array([ self.env.step(action) for data in np.arange(self.skip_n)])
        

        new_obs = skips[:,0]
        rewards = skips[:,1]
        dones = skips[:,2]
        skips = None 


        return (new_obs.sum(axis=0),rewards.sum(),dones.sum(),{})
    
    def env_step(self,action: np.ndarray):
        self.step += 1
        
        self.skip_n = 4
        new_obs,reward,done,info,infoX = self.env.step(action)

        # self.env.render('rg')

        cv2.imwrite(f'x{self.gpu}.png',new_obs*255)

        # self.eps_index += 1
        self.reward_total_actual += reward

        reward = reward - 0.1

        self.reward_total += reward
        new_obs = new_obs/255
        self.obs = new_obs
        
        if (self.step == self.max_steps):
            done = True

        self.done = done


        reward_returning = self.reward_function(self.reward_total)

        return (new_obs.astype(np.float32),np.array(reward_returning *  (self.discount_factor**self.epoch)).astype(np.float32),np.array(done).astype(np.int32))
    
    
    def env_step_tf(self,action):
        new_obs,reward,done = tf.numpy_function(self.env_step,[action],[tf.float32,tf.float32,tf.int32])
        

        return (tf.image.resize(tf.reshape(new_obs,self.obs_shape),self.obs_resized_shape),tf.reshape(reward,[1])[0],tf.reshape(done,[1])[0])
    
    

    
    def greedy_action_tf(self,action_probs):
        
        return tf.numpy_function(self.greedy_action,[action_probs],[tf.float32])
        

    
    def train_episode(self):
        
        obs = self.env_step_tf_reset()
        
        
        rewards = tf.TensorArray(size=0,dynamic_size=True,dtype=tf.float32)
        actions = tf.TensorArray(size=0,dynamic_size=True,dtype=tf.float32)
        dones = tf.TensorArray(size=0,dtype=tf.int32,dynamic_size=True)
        v_sX = tf.TensorArray(size=0,dtype=tf.float32,dynamic_size=True)
    
        v_sX_new = tf.TensorArray(size=0,dtype=tf.float32,dynamic_size=True)
        actions_new = tf.TensorArray(size=0,dtype=tf.float32,dynamic_size=True)
        

        reward_total = tf.cast(0.0,dtype=tf.float32)
        # action_size_expanded = tf.expand_dims(self.action_size,axis=0)

        for data in tf.range(self.n_step):
            
            
            action,v_s = self(tf.expand_dims(obs,axis=0))

            # action = uniform_eps_exp_var(action,self.get_eps_tf(),action_size_expanded)
            # action = action

            obs,reward,done = self.env_step_tf(tf.argmax(action,axis=-1)[0])

            reward_total += reward
            
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
    
    
    
    def did_nan(self,loss_x: np.ndarray):
        self.did_nan_cond = loss_x
    
    def did_nan_tf(self,loss_x):
        return tf.numpy_function(self.did_nan,[tf.math.is_nan(tf.reduce_mean(loss_x))],[])
    
    def get_eps(self):
        # self.eps = self.eps_array[self.eps_index]
        return np.array(self.eps).astype(np.float32) 


    def get_eps_tf(self):
        return tf.reshape(tf.numpy_function(self.get_eps,[],[tf.float32]),[1])[0]




    def compute_loss(self,actions,rewards,dones_mask,v_sX,v_sX_new,actions_new):
        
        # Both be equal to (None)

        actions = tf.reshape(tf.cast(actions,dtype=tf.float32),[-1,self.action_size])
        rewards = tf.expand_dims(tf.cast(rewards,dtype=tf.float32),axis=-1)
        dones_mask = tf.expand_dims(tf.cast(dones_mask,dtype=tf.float32),axis=-1)
        v_sX = tf.expand_dims(tf.cast(tf.reshape(v_sX,[-1]),dtype=tf.float32),axis=-1)
        v_sX_new = tf.expand_dims(tf.cast(tf.reshape(v_sX_new,[-1]),dtype=tf.float32),axis=-1)
        actions_new = tf.cast(tf.reshape(actions_new,[-1,self.action_size]),dtype=tf.float32)
        
        
        

        
        # Other inti vairables. 
        x_coffe = tf.cast(0.1,dtype=tf.float32)
        y_coffe = tf.cast(0.5,dtype=tf.float32) 
        z_coffe = tf.cast(10,dtype=tf.float32) 
        gamma = tf.cast(0.98,dtype=tf.float32)    
        
 

        # loss_x = tf.cast(tf.math.lgamma(actions_new) * rewards,dtype=tf.float32) * tf.math.exp(actions*v_sX))
        


        # *(((1.0 - self.get_eps_tf())*0.75)+0.25)
        action_loss = tf.math.lgamma(tf.where(tf.math.is_inf(tf.divide(1.0,actions_new)),actions_new+1e6,actions_new)) 
        self.check_is_nan_cond_tf(action_loss,actions_new,'action_loss')


        exp_value = custom_exp_lower_base(actions)
        self.check_is_nan_cond_tf(exp_value,actions,'exp_value')


        # GPU-0
        loss_x = (tf.cast(action_loss * v_sX_new,dtype=tf.float32) * ((rewards * exp_value )*v_sX)/10)

        
        
        

        # GPU-1
        # loss_x = tf.cast(tf.math.lgamma(actions_new) * v_sX_new,dtype=tf.float32) * (rewards * (tf.math.exp(actions) / tf.reduce_mean(tf.math.exp(actions)) )*v_sX) / 10






        # loss_x = tf.cast(tf.math.lgamma(actions_new) * v_sX_new,dtype=tf.float32) * (rewards * tf.math.log(actions*v_sX))

        # loss_x = tf.cast(tf.math.lgamma(actions_new) * rewards,dtype=tf.float32) * (rewards * tf.math.exp(actions*v_sX))

        # FORMULA GOES HERE. #$ 
        

        
        
        #$ FORMULA GOES HERE.
        
    
        # loss_new = 
            
        
        


        
        return loss_x