




import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


from weights_addition_converter import weights_addition_converter




class proccess_functions(tf.keras.models.Model):
    
    def __init__(self,*args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        
        self.squared_values = 2**np.arange(4,10240)
        self.crop_size = [128,128]
        self.weights_index = 5
        
        self.weight_image_resize_shape = [128,128]

    
    
    def handel_proccess_function(self,key):
        
        return {
            'add_net': self.processing_function_add_net_tf,
            'crop_net_pre': self.proccessing_function_crop_net_tf,
            'c_net': self.processing_function_c_net_tf,
            'video_net': self.mask_out_frames_and_convert_tf,
            }[key]

        
        
    def processing_function_add_net(self,x):
        c_net_weights = self.model_c_net.get_weights()
        self.model_c_net = self.model_c_inst.define_model(c_net_weights[0].shape[0])
        self.model_c_net.set_weights(weights_addition_converter(2).step(x,weights=c_net_weights))

    def processing_function_add_net_tf(self,x):
        return tf.numpy_function(self.processing_function_add_net,[x],[])
    
    
    def find_nearest(self,data):
        
        return self.squared_values[np.argmin(np.abs(self.squared_values - data))]
        
    def proccessing_function_crop_net(self,x):
        
        if (self.squared_values[self.squared_values == x.shape[1]].size == 0 or self.squared_values[self.squared_values == x.shape[2]]):
            
            return (np.array([self.find_nearest(x.shape[1]),self.find_nearest(x.shape[2])]).astype(np.int32),np.array(True).astype(np.int32))
        
        return ([x.shape[1],x.shape[2]],np.array(False).astype(np.int32)) 
            
            
    
    def proccessing_function_crop_net_tf(self,x):
        
        
        shape_resize,resize_cond = tf.numpy_function(self.proccessing_function_crop_net,[x],[tf.int32,tf.int32])

        if tf.cast(resize_cond,dtype=tf.bool):
            x = tf.image.resize(x,shape_resize)
        
        
        return x 
        
    # output: [None,resize_shape[0],resize_shape[1],weighted_filters_applied]
    def processing_function_c_net_tf(self,features,bboxes,objects):
        
        
        infomation = tf.gather(bboxes,tf.where(objects == 1))
        
        return tf.image.crop_and_resize(features,infomation,tf.cast(tf.zeros(tf.shape(infomation)[0]),dtype=tf.int32),self.crop_size)
        
        
    def weight_multiplication_formula(self):
        
        # TEST HERE
        weights = self.model.get_weights()
        
        y_x = np.arange(2048)**2
        x = (np.hstack([ data.reshape(-1)  for data in (weights[:self.weights_index-1] + weights[self.weights_index:]) ] ))
        amount = y_x[np.abs(y_x - x.shape[0]).argmin()-1]
        int_squared = int(np.sqrt(amount))
        x = x[:amount]
        
        return (x.reshape(1,int_squared,int_squared,1).astype(np.float32),int_squared)
    
    @tf.function()
    def processing_function_dpf_net_tf(self):
        weights,int_squared = tf.numpy_function(self.weight_multiplication_formula,[],[tf.float32,tf.int32])
        int_squared = tf.reshape(int_squared,[1])[0]
        return tf.image.resize(tf.reshape(weights,[1,int_squared,int_squared,1]),self.weight_image_resize_shape)
    
    
    @tf.function()    
    def cnet_transformations_tf(self,featured_path_images,classes_node_found):
        

        return tf.reshape(tf.concat([tf.reshape(featured_path_images,[-1]),classes_node_found],axis=1),[-1,129,128,1])
        
        
        
        

    
    @tf.function()
    def mask_out_frames_and_convert_tf(self,frames,actions):

        x = tf.gather(frames,tf.where(tf.argmax(actions,axis=-1) == 1))
        return x
        



        
            
    
    