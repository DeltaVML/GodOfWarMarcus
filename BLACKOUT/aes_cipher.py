


import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
import tensorflow as tf 
import tensorflow_io as tfio
import numpy as np 

# from speak_to_me_env import speak_to_me_env

import matplotlib.pyplot as plt 

class aes_cipher(object):

    def __init__(self, key , input_shape=[204,204]):
        self.bs = AES.block_size
        self.key = hashlib.sha256(key.encode()).digest()
        self.image_shape = input_shape
        # self.enc_shape = [224,184]
        self.enc_shape = [428,389]
        self.found_best_shape = np.array([])
        
        
        
        
        # self.vocab_size = 65j
        # self.vocab = np.load('vocab.npy')

    def encrypt(self, raw):
        raw = self._pad(raw)
        # Possible option to remove Random new -- M NOTE
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)

        return base64.b64encode(iv + cipher.encrypt(raw.encode()))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]
    

    # is taking a int value 
    # returning, a (None,None,end_dims)*prod.
    
    def pad_hunna(self,value):
        
        return ''.join([ '00'+data if len(data) == 1 else ('0'+data if len(data) == 2 else data) for data in value.astype(np.str_).reshape(-1)])
    
    def tokenize(self,value):

        rounded_value = self.pad_hunna(value)
        # rounded_value = ''.join([ '00'+data if len(data) == 1 else ('0'+data if len(data) == 2 else data) for data in value.astype(np.str_).reshape(-1)])

        # SO FAR bytes values, of the value.


        enc_value = self.encrypt(rounded_value)

        self.enc_pad_value = enc_value.decode('utf-8')[:AES.block_size]
        
        model_input = np.array([ data for data in enc_value.decode('utf-8')[AES.block_size:]]).reshape(self.enc_shape)

        return model_input
    
    
    def pokenize_outputs(self,num):
        
        x = num / np.arange(50,600)
        x = x[x - x.astype(np.int32) == 0.0]
        
        print(num / x)
        print(x)
        # return (num / x,x)
        
        
    
    def test_tokenize(self,value):
        rounded_value = self.pad_hunna(value)
        # rounded_value = ''.join([ '00'+data if len(data) == 1 else ('0'+data if len(data) == 2 else data) for data in value.astype(np.str_).reshape(-1)])

        # SO FAR bytes values, of the value.


        enc_value = self.encrypt(rounded_value)

        self.enc_pad_value = enc_value.decode('utf-8')[:AES.block_size]
        
        
        
        model_input = np.array([ data for data in enc_value.decode('utf-8')[AES.block_size:]]).reshape(self.enc_shape)

        return model_input

 
    
    def de_tokenize(self,model_input):

    
        encypted_revered_model_output =  (self.enc_pad_value+ ''.join(model_input.reshape(-1))).encode('utf-8')
        
        
        decrypted_value = self.decrypt(encypted_revered_model_output)

        de_tokenized_value = np.array([  decrypted_value[window_index*3:(window_index+1)*3]  for window_index in np.arange(len(decrypted_value)//3)]).astype(np.int32).reshape(self.image_shape)

        
        return de_tokenized_value
    
    def find_best_shape(self,size):
        
        
        range_x = np.arange(1024)
        x = np.array([ size / data for data in range_x])
        data_x = (x - x.astype(np.int32)) == 0
        x = x[data_x]
        factors = range_x[data_x]
        
        print(x)
        print(factors)
        
        


        for data in np.arange(factors.size):

            width,height = (x[data],factors[data])


            if (width %2 == 0 and height %2 == 0):
                
                
                
                

                try:
                    min_x = np.abs(self.found_best_shape[:,2].min() + self.found_best_shape[:,1].min())
                    min_cond = min_x > np.abs(width - height)
                except:
                    min_cond = True 
                    
                
                if (min_cond):
                    self.found_best_shape = np.append(self.found_best_shape,[self.image_shape[0],width,height]).reshape(-1,3)


        
        return data_x
        
        
        
    
    @tf.function()
    def audio_converstion(self,audio_test=None,rate=None,file_audio=None):
        
            
        if (audio_test == None):
            audio_test = tfio.audio.AudioIOTensor(file_audio)
            audio_test,rate = (audio_test.to_tensor(),audio_test.rate)
        

        spectrogram = tfio.audio.spectrogram(audio_test[:,0], nfft=1024, window=512, stride=128)
        mel = tf.image.resize(tf.expand_dims(tfio.audio.melscale(spectrogram, rate=rate, mels=128, fmin=0, fmax=1000),axis=-1),self.image_shape)
        mel = tf.cast((mel / 1000)*999,dtype=tf.int32)

        return mel
    
    
            
    def pad_string(self,data):
        data = np.array([ord(data) for data in data])
        data = np.append(np.repeat(ord(' '),(self.image_shape[0]*self.image_shape[1])-data.size),data)
        
      
        return data
        


if (__name__ == '__main__'):
    
    # x = plt.imread('main.jpeg')
    
    # # audio_test = tfio.audio.decode_mp3(tf.io.read_file('./main.mp3'))


    text = "You're mother"
    

    # x = speak_to_me_env().handel_step_input()


    # wanker = np.array([])
    # for data in np.arange(2000):
    ciper = aes_cipher(str(''),[204,204])
    import pdb; pdb.set_trace()

    # ciper.pad_string()
    # for data in np.arange(128)+128:
    #     ciper.image_shape = [data,data]
    
    mel = ciper.audio_converstion(x,rate=16000,file_audio='').numpy()

    # ciper.test_tokenize(mel)
    token = ciper.tokenize(mel)
    de_token = ciper.de_tokenize(token)    

    print(((de_token == mel.reshape(204,204)).sum() == mel.size))
    print() 
    
    


