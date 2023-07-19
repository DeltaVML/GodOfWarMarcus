


import tensorflow as tf 



class custom_callback(tf.keras.callbacks.Callback):


    def __init__(self,*args, **kwargs):

        super().__init__(*args, **kwargs)

    def on_batch_end(self,batch,logs=None):

        if (self.model.done):
            

            print(self.model.reward_total_actutal)
            # print('RewardTotalAccuracy (ACC)',(self.model.reward_total_actutal / 300).mean())