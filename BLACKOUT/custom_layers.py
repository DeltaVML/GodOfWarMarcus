


import tensorflow as tf 
from tensorflow.keras.layers import Embedding,Dense


class AddOneToLayer(tf.keras.layers.Layer):


    def __init__(self,*args,**kwargs):

        super().__init__()

    def call(self,x):
        return tf.expand_dims(x,axis=-1)



class NuronExchange(tf.keras.layers.Layer):


    def __init__(self,seq_len,nuron_exchange,*args,**kwargs):
        
        super().__init__(*args,**kwargs)

        self.d1 = Dense(1,activation='relu')
        self.d2 = Dense(seq_len,activation='relu')
        self.d3 = Dense(nuron_exchange*seq_len,activation='relu')
    
    @tf.function()
    def rescaled(self,value):

        return tf.math.divide_no_nan(value,tf.reduce_max(value))

    def call(self,x):
        
        
        x_new = x + tf.squeeze(tf.expand_dims(x,axis=0),axis=-1)
        return self.d3(self.d2(x_new) * self.rescaled(self.d1(x)))





class PosEmbNE(tf.keras.layers.Layer):


    def __init__(self,seq_len,nuron_exchange,vocab_size,*args,**kwargs):

        super().__init__(*args,**kwargs)
        self.layer = NuronExchange(seq_len,nuron_exchange)
        self.d_token = tf.cast(tf.constant(seq_len*nuron_exchange),dtype=tf.float32)

        self.ten_tho = tf.cast(tf.constant(10000),dtype=tf.float32)
        self.two_x = tf.cast(tf.constant(2),dtype=tf.float32)

        self.vocab_size = tf.cast(tf.constant(vocab_size),dtype=tf.float32)

    @tf.function()
    def find_pos(self,x):

        return tf.math.divide_no_nan(x,tf.pow(self.ten_tho,tf.divide(tf.multiply(self.two_x,tf.expand_dims(tf.expand_dims(tf.cast(tf.range(tf.shape(x)[1]),dtype=tf.float32),axis=-1),axis=0)),self.d_token)))

    # def compute_mask(self,*args,**kwargs):

    #     mask = self.layer.compute_mask(*args,**kwargs)

    #     if (mask is not None):

    #         mask = mask[:,None,:] & mask[:,:,None]

    #     return mask 


    def call(self,x):
        x = self.layer(x)

        return tf.sin(self.find_pos(x))
    


class Concat(tf.keras.layers.Layer):
    
    
    def __init__(self,*args, **kwargs):
        
        super().__init__(*args, **kwargs)

    def call(self,inputs):
        return tf.concat(inputs,axis=1)



    


class Sep(tf.keras.layers.Layer):
    
    
    def __init__(self,*args, **kwargs):
        
        super().__init__(*args, **kwargs)

    def call(self,input_x):
        
        return (tf.expand_dims(input_x[:,0],axis=1),tf.expand_dims(input_x[:,1],axis=1),tf.expand_dims(input_x[:,2],axis=1))
    
    
class Expand(tf.keras.layers.Layer):
    
    def __init__(self,axis=1,*args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.axis = axis
    def call(self,x):
        return tf.expand_dims(x,axis=self.axis)


class Transition(tf.keras.layers.Layer):
    
    def __init__(self,*args,**kwargs):
        
        super().__init__(*args,**kwargs)
        
        
    def call(self,data):
        
        x,mask = tuple(data)
        
        
        if (mask is not None):
            
            x = x*mask

        
        return x


class PosEmb(tf.keras.layers.Layer):


    def __init__(self,d_token,vocab_size,*args,**kwargs):

        super().__init__(*args,**kwargs)
        self.layer = Embedding(vocab_size,d_token)
        self.d_token = tf.cast(tf.constant(d_token),dtype=tf.float32)

        self.ten_tho = tf.cast(tf.constant(10000),dtype=tf.float32)
        self.two_x = tf.cast(tf.constant(2),dtype=tf.float32)

        self.vocab_size = tf.cast(tf.constant(vocab_size),dtype=tf.float32)

    @tf.function()
    def find_pos(self,x):

        return tf.math.divide_no_nan(x,tf.pow(self.ten_tho,tf.divide(tf.multiply(self.two_x,tf.expand_dims(tf.expand_dims(tf.cast(tf.range(tf.shape(x)[1]),dtype=tf.float32),axis=-1),axis=0)),self.d_token)))

    def compute_mask(self,*args,**kwargs):

        mask = self.layer.compute_mask(*args,**kwargs)

        if (mask is not None):

            mask = mask[:,None,:] & mask[:,:,None]

        return mask 


    def call(self,x_input):
        x = self.layer(x_input)


        return tf.sin(self.find_pos(x))
    


class Slicing(tf.keras.layers.Layer):


    def __init__(self,size_n,*args, **kwargs):

        super().__init__(*args, **kwargs)

        self.size_n = size_n

    def call(self,data):


        return data[:,:self.size_n]


