

import base64
import tensorflow as tf 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import pandas as pd 

import json 

from aes_cipher import aes_cipher

import re 


class processing:
    
    
    def __init__(self):
        
        
        # self.datasets = pd.Dataframe([],columns=['text','key']) 
    

        self.hacks = os.listdir('./data')
        self.hack = os.listdir('./enwiki20201020')
        
        self.created_data = np.array([])
        self.window_size = 32
        self.key_index = 0 
        
        self.key_file = 0
        self.key_quota = 0
        self.data_index = 13000
        self.divide_and_conq_max = 100
        self.key_amount = 20
        self.recusive_index = 0
        self.total_values = 0
        self.keys = np.array([ './data/'+data for data in os.listdir('./data')])
        
        
        self.new_dir = '/media/barbra/44763AD6763AC886/main'
        
        self.current_keys = None 
        self.current_key = None 
        
    
        re_string = ''
        
        for data in np.array([ chr(data) for data in np.arange(1000)])[32:127]:
            re_string += data
            
        self.re_string = '[^'+re_string+'\s*]'
    
        self.current_dir = f'{self.new_dir}/{self.data_index//1000}'
    
    def read_to_numpy(self,file):
        

        with open(file,'r') as read_data: x = np.array(read_data.readlines())
        return x        
    
    
    def convert_data_to_understandable(self,data):
        
        
        try:
            return re.sub('\s+'," ",re.sub(self.re_string,'',data))
        except:
            import pdb; pdb.set_trace()
    
    def generate_keys(self):
        
        
        
        self.key_quota += 1



        try:
            
            if (self.current_key == None):
                raise "Wanker"
            
            if (self.key_quota == self.key_amount):
                self.key_quota = 0
                self.key_index += 1 
                
                if (self.key_index == self.current_keys.size-1 ):
                    self.key_index = 0
                    self.key_file += 1
                    
                    if (self.key_file == self.keys.size-1):
                        self.key_file = 0
                        
                    
                    self.current_keys = self.read_to_numpy(self.keys[self.key_file])
                    
                    
                
                self.current_key = self.current_keys[self.key_index][:-1]


            
        
        
        except Exception as e:
            print(e)
            self.current_keys = self.read_to_numpy(self.keys[self.key_file])


            self.current_key = self.current_keys[self.key_index][:-1]    



        

        
           
            
            
    def write_data(self):
        
        
        print(self.data_index)
        
        if (self.data_index %1000 == 0):
            self.current_dir = f'{self.new_dir}/{self.data_index//1000}'
            os.mkdir(self.current_dir)  
            

        np.save(f'{self.current_dir}/{self.data_index}.npy',self.created_data)
        self.data_index += 1
        self.created_data = np.array([])

    
    
    def divide_and_conq_max_func(self,data):
        

        
        cond_divide = True 
        index = 0
        self.recusive_index = 0
        while cond_divide:
            data,index,cond_divide = self.update_create(data,index=index)
            self.recusive_index = 0

            if ((self.created_data.size > 3000*3)):
                self.write_data()
    
    def update_create(self,data,index=0):
        
        # print(self.total_values)        
        self.total_values += 1


    
        
        windowed_data = data[(self.window_size*index):(self.window_size * (index+1))]
        
        
        self.generate_keys()
        data = data[32:]

        windowed_data_enc = aes_cipher(self.current_key)
        # windowed_data_enc.decode('utf-8').encode('utf-8') == windowed_data_enc
        
        
            

        
        try:        
            windowed_data_enc = windowed_data_enc.encrypt(windowed_data).decode('utf-8') 
        except:
            import pdb; pdb.set_trace()
            
            
        
        self.created_data = np.append(self.created_data,[self.current_key,windowed_data,windowed_data_enc])
        

        

        cond_window = len(windowed_data) != 0
        if (self.recusive_index >= self.divide_and_conq_max and cond_window):
            rec = True 
            return (data,index,rec)
            
        

                
        if (cond_window):
            self.recusive_index += 1
            data,index,rec = self.update_create(data,index+1)
        else:
            rec = False   

            

            
        
        return (data,index,rec)
        
        
    
    def main(self):
        
        
        

        for data in np.array(self.hack):
            with open('./enwiki20201020/'+data,'r') as read_data: x = [ self.divide_and_conq_max_func(self.convert_data_to_understandable(data['text'])) for data in json.load(read_data)]
            

            
            
if (__name__ == '__main__'):
    
    
    processing().main()

                
                

        

            
        