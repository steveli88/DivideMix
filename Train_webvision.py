from __future__ import print_function
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from InceptionResNetV2 import *
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet


# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t Labeled loss: %.2f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)   
        
        #penalty = conf_penalty(outputs)
        L = loss #+ penalty      

        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t CE-loss: %.4f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        
        
def test(epoch,net1,net2,test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)                 
            acc_meter.add(outputs,targets)
    accs = acc_meter.value()
    return accs


def eval_train(model,all_loss):    
    model.eval()
    num_iter = (len(eval_loader.dataset)//eval_loader.batch_size)+1
    losses = torch.zeros(len(eval_loader.dataset))    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]       
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' %(batch_idx,num_iter)) 
            sys.stdout.flush()    
                                    
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # fit a two-component GMM to the loss
    input_loss = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = InceptionResNetV2(num_classes=args.num_class)
    model = model.cuda()
    return model


def sort_dict(myDict):
    myKeys = list(myDict.keys())
    myKeys.sort()
    sorted_dict = {i: myDict[i] for i in myKeys}
    return sorted_dict


def label_stats(noisy_label, true_label, epoch, log):
    label_stats = {}
    correct_label_stats = {}
    correct_label = 0
    for i in range(len(noisy_label)):

        if noisy_label[i] in label_stats:
            label_stats[noisy_label[i]] += 1
        else:
            label_stats[noisy_label[i]] = 1

        if noisy_label[i] == true_label[i]:
            correct_label += 1
            if noisy_label[i] in correct_label_stats:
                correct_label_stats[noisy_label[i]] += 1
            else:
                correct_label_stats[noisy_label[i]] = 1

    label_stats = sort_dict(label_stats)
    correct_label_stats = sort_dict(correct_label_stats)

    log.write('Epoch %d \n' % epoch)
    log.write('Number of labels for classes: %s \n' % label_stats)
    log.write('Correct labels for classes: %s \n' % correct_label_stats)
    log.write('Overall accuracy: %.2f \n' % (correct_label / len(noisy_label)))

    log.write('Total sample selected: %.2f \n' % (sum(label_stats.values())))
    log.write('Total clean sample selected: %.2f \n' % (sum(correct_label_stats.values())))
    # for key in correct_label_stats:
    #     log.write('The Precision of Class %d is %.2f \n' % (key, correct_label_stats[key] / label_stats[key]))

    log.flush()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
    parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
    parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--id', default='', type=str)
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--num_class', default=50, type=int)
    parser.add_argument('--data_path', default='/home/lorentz/Project/Datasets/miniwebvision/', type=str, help='path to dataset')

    parser.add_argument('--cluster_prior_epoch', default=100, type=int)
    parser.add_argument("--cluster_file", default='features_clusters_webvision_dinov2_vitl14_reg_f1024_c1000.pt', type=str)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    curr_time = time.strftime("%m%d%H%M", time.localtime())
    directory = os.path.join('checkpoint', f'{curr_time}_webvision')
    if not os.path.exists(directory):
        os.makedirs(directory)

    stats_name = f'webvision_{args.num_epochs}_stats_{curr_time}.txt'
    test_name = f'webvision_{args.num_epochs}_acc_{curr_time}.txt'
    stats_log = open(os.path.join(directory, stats_name), 'w')
    test_log = open(os.path.join(directory, test_name), 'w')
    test_log.write(str(args) + '\n')

    # todo need to upload the clustering file
    cluster_file = args.cluster_file
    n_clusters = 1000 # todo change cluster number according to needs
    test_log.write(cluster_file + '\n')

    warm_up=1

    loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_workers=5,root_dir=args.data_path,log=stats_log, num_class=args.num_class)

    print('| Building net')
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy()

    all_loss = [[],[]] # save the history of losses from two networks
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)

    for epoch in range(args.num_epochs+1):
        lr=args.lr
        if epoch >= 50:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr

        eval_loader = loader.run('eval_train')
        web_valloader = loader.run('test')
        imagenet_valloader = loader.run('imagenet')
        warmup_trainloader = loader.run('warmup')

        if epoch < warm_up:
            warmup_trainloader = loader.run("warmup")
            print("Warmup Net1")
            warmup(epoch, net1, optimizer1, warmup_trainloader)
            print("\nWarmup Net2")
            warmup(epoch, net2, optimizer2, warmup_trainloader)

        else:
            prob1, all_loss[0] = eval_train(net1, all_loss[0])
            prob2, all_loss[1] = eval_train(net2, all_loss[1])

            pred1 = prob1 > args.p_threshold
            pred2 = prob2 > args.p_threshold

            if epoch <= args.cluster_prior_epoch:
                low_loss_idx_1 = torch.tensor(prob1 > args.p_threshold).cuda()
                low_loss_idx_2 = torch.tensor(prob2 > args.p_threshold).cuda()
                expanded_low_loss_idx_1 = torch.clone(low_loss_idx_1)
                expanded_low_loss_idx_2 = torch.clone(low_loss_idx_2)
                # expand correct label via clustering
                cls = torch.load(cluster_file)
                noisy_labels_tensor = torch.tensor(warmup_trainloader.dataset.noise_label).cuda()
                for i in range(n_clusters):
                    correct_labels_1 = torch.masked_select(noisy_labels_tensor[cls[i]['idx']],
                                                           low_loss_idx_1[cls[i][
                                                               'idx']])  # what are labels of low loss samples?
                    expanded_low_loss_1 = torch.isin(noisy_labels_tensor[cls[i]['idx']], correct_labels_1) + \
                                          low_loss_idx_1[cls[i][
                                              'idx']]  # cls[i]['idx'] same cluster samples; torch.isin match low loss label
                    expanded_low_loss_idx_1[cls[i]['idx']] = expanded_low_loss_1  #

                    correct_labels_2 = torch.masked_select(noisy_labels_tensor[cls[i]['idx']],
                                                           low_loss_idx_2[cls[i]['idx']])
                    expanded_low_loss_2 = torch.isin(noisy_labels_tensor[cls[i]['idx']], correct_labels_2) + \
                                          low_loss_idx_2[cls[i]['idx']]
                    expanded_low_loss_idx_2[cls[i]['idx']] = expanded_low_loss_2

                prob1 = (expanded_low_loss_idx_1 * 1.).cpu().numpy()
                prob2 = (expanded_low_loss_idx_2 * 1.).cpu().numpy()

                pred1 = (prob1 > args.p_threshold)
                pred2 = (prob2 > args.p_threshold)

            print("Train Net1")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred2, prob2
            )  # co-divide
            train(
                epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader
            )  # train net1
            # stats_log.write('Low loss labels from Model 2 to Model 1\n')
            # label_stats(labeled_trainloader.dataset.noise_label, labeled_trainloader.dataset.clean_label, epoch, stats_log)

            print("\nTrain Net2")
            labeled_trainloader, unlabeled_trainloader = loader.run(
                "train", pred1, prob1
            )  # co-divide
            train(
                epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader
            )  # train net2
            # stats_log.write('Low loss labels from Model 1 to Model 2\n')
            # label_stats(labeled_trainloader.dataset.noise_label, labeled_trainloader.dataset.clean_label, epoch, stats_log)

            stats_log.write('\n')

        web_acc = test(epoch,net1,net2,web_valloader)
        imagenet_acc = test(epoch,net1,net2,imagenet_valloader)

        print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
        test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
        test_log.flush()

        # print('\n==== net 1 evaluate training data loss ====')
        # prob1,all_loss[0]=eval_train(net1,all_loss[0])
        # print('\n==== net 2 evaluate training data loss ====')
        # prob2,all_loss[1]=eval_train(net2,all_loss[1])
        # torch.save(all_loss,'./checkpoint/%s.pth.tar'%(args.id))

