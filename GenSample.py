from HyperTools import *
import argparse
import os

def main(args):  
    dataID = args.dataID
    X,Y,train_array,test_array = LoadHSI(dataID,args.train_ratio)

    if dataID==1:
        save_pre_dir = './Data/PaviaU/'
    elif dataID==2:
        save_pre_dir = './Data/Salinas/'
    elif dataID==3:
        save_pre_dir = './Data/Indian_pines/'

    Y -= 1

    if os.path.exists(save_pre_dir)==False:
        os.makedirs(save_pre_dir)

    
    np.save(save_pre_dir+'X.npy',X)
    np.save(save_pre_dir+'Y.npy',Y)
    np.save(save_pre_dir+'train_array.npy',train_array)
    np.save(save_pre_dir+'test_array.npy',test_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--train_ratio', type=int, default=0.2)

    main(parser.parse_args())