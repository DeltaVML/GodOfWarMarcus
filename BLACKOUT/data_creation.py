


# from aes_cipher import aes_cipher

import numpy as np 

# from collections import permutations

class data_creation:
    
    
    def __init__(self):
        
        self.max_length = 256
        self.index_gen = 0

        self.word_vocab = np.load('vocab.npy')

        self.charcters = np.array([ chr(data) for data in np.arange(256)])
        
        self.index_key = 10
        
        self.dob_terms = np.unique(np.append(np.arange(10).astype(np.str_),[ '0'+data if (len(data) == 1) else data for data in np.arange(99).astype(np.str_)]))        
        # x = np.arange(99)



        
    def shiftWord(self,word):
        
        len_word = len(word)
        shiftWord = np.random.randint(1,len_word+1)
        shiftMetaWord = np.random.randint(0,4+1)
        

        new_word = word[:shiftWord] + self.gen_ran(14) + word[shiftWord:]        
        

        
        if (np.random.rand(0,2)):
            return new_word[:-shiftMetaWord]
        else:
            return new_word[shiftMetaWord:]
        
        return new_word


        # for data in np.arange(word):
        
    def generate_dob_ext(self):
        
        rans = np.random.randint(0,self.dob_terms.size,2)
        
        
        return self.dob_terms[rans[0]] + self.dob_terms[rans[1]]
    
    

        
        
    
    def create_keys(self):
        
        
        
        
        y = np.array([ chr(data) for data in np.arange(280000)])
        
            
    def gen_ran(self,length,disable_log=False,could_be=True):
        
        
        if (could_be and np.random.randint(0,8) == 1):
            return self.generate_dob_ext()
            
        else:
            if (disable_log):
                self.index_gen += 1
                print(self.index_gen)      

            x = ''.join(self.charcters[np.random.randint(0,self.charcters.size,length)].astype(np.str_))

        return x
    

        
        
    
    def genRanWord(self,length=20):
        self.index_gen += 1

        words = []
        
        
        words = self.word_vocab[np.random.randint(0,self.word_vocab.size,np.random.randint(1,length+1,1))]
        titleOrdCaps = np.random.randint(0,3+1,words.size)

        x = np.array([ words[index].upper() if titleOrdCaps[index] == 2 else  (words[index].title() if titleOrdCaps[index] == 1 else words[index].lower())   for index in np.arange(words.size)])
        
        shiftyingX = np.random.randint(0,4,words.size+1)
        x = np.array([ self.shiftWord(x[data]) if shiftyingX[data] == 1 else x[data] for data in np.arange(words.size)])
        
        main_word = ''.join(x)
        
        print(self.index_gen)

        return main_word
    
    
    def write_to_dataset(self,x):
        
        with open(f'./data/{self.index_key}keys.txt','w',encoding='utf-8') as write_data: write_data.writelines(x)
        self.index_key += 1
        
    def generate_worker(self):
        
        self.data_keys = np.array([])
        for data in np.arange(100):
            x = np.array([ self.gen_ran(4096,disable_log=True,could_be=False) if (data %5 == 0) else self.genRanWord(np.random.randint(1,21)) for data in np.arange(int(5e5))])

            import pdb; pdb.set_trace()
            self.write_to_dataset(x)


        import pdb; pdb.set_trace()
        return x 
    
    def main(self):
        
        
        aes_cipher('me hello wanker.com').encrypt('asdadsdasasddddddddddddddddddddddapKDp[kp[wqrpwepkr[ewrp[pkrew[pwqerkpwrekp[werpk[wera:DSMopakodpffewaW{KFDA[pkewr[wek[r[pewraW{LDp[awd[pkewrpkpewrkp[ewr')
        
if (__name__ == '__main__'):
    
    data_creation().generate_worker()