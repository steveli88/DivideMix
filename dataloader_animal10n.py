from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

            
def unpickle(file):
    # import _pickle as cPickle
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class animal10n_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log=''): 
        assert dataset == 'animal10n'
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise

        self.nb_classes = 10

        self.train_folder = os.path.join(root_dir, 'training')
        self.test_folder = os.path.join(root_dir, 'testing')
        self.train_files = os.listdir(self.train_folder)
        self.test_files = os.listdir(self.test_folder)

        if self.mode=='test':
            self.test_label = [int(i.split('_')[0]) for i in self.test_files]
        else:
            train_files = [i for i in self.train_files]
            train_label = noise_label = [int(i.split('_')[0]) for i in self.train_files]
            
            if self.mode == 'all':
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_files = [train_files[i] for i in pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))
                
                self.clean_label = [train_label[i] for i in pred_idx]
                self.clean_label_idx = pred_idx        
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_file, target, prob = self.train_files[index], self.noise_label[index], self.probability[index]
            img = np.asarray(Image.open(os.path.join(self.train_folder, img_file)))
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img_file = self.train_files[index]
            img = np.asarray(Image.open(os.path.join(self.train_folder, img_file)))
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img_file, target = self.train_files[index], self.noise_label[index]
            img = np.asarray(Image.open(os.path.join(self.train_folder, img_file)))
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img_file, target = self.test_files[index], self.test_label[index]
            img = np.asarray(Image.open(os.path.join(self.test_folder, img_file)))
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_files)
        else:
            return len(self.test_files)
        
        
class animal10n_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file

        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = animal10n_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = animal10n_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = animal10n_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = animal10n_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = animal10n_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        