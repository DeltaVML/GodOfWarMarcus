

import tensorflow as tf 

import numpy as np 

@tf.function()
def custom_exp_lower_base(x):

    return 1.218**x




@tf.function()
def uniform_eps_exp_var(data,eps,action_size):

    # print(1 * ((eps*0.5)+0.5 * 2) )
    return ((((tf.expand_dims(tf.random.uniform(action_size),axis=0)*2)-1)*eps) + data ) / (1 * ((eps*0.5)+0.5 * 2))
    # return ((((tf.random.uniform(action_size)*2)-1)*eps) + data) / (2/eps)



# current_logits = np.linspace(0,1,1000)[np.random.randint(0,1000,4)]

# for data in np.linspace(0,1,1000)[::-1]:

#     print(uniform_eps_exp_var(current_logits,data,np.expand_dims(4,axis=0)).numpy(),data,current_logits)
