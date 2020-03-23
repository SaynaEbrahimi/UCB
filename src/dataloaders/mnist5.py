import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms

########################################################################################################################

def get(data_path,seed,fixed_order=False,pc_valid=0):
    data={}
    taskcla=[]
    size=[1,28,28]

    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}
    dat['train']=datasets.MNIST(data_path,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.MNIST(data_path,train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    data[0]={}
    data[0]['name']='mnist-0-1'
    data[0]['ncla']=2
    data[1]={}
    data[1]['name']='mnist-2-3'
    data[1]['ncla']=2
    data[2]={}
    data[2]['name']='mnist-4-5'
    data[2]['ncla']=2
    data[3]={}
    data[3]['name']='mnist-6-7'
    data[3]['ncla']=2
    data[4]={}
    data[4]['name']='mnist-8-9'
    data[4]['ncla']=2
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        data[0][s]={'x': [],'y': []}
        data[1][s]={'x': [],'y': []}
        data[2][s]={'x': [],'y': []}
        data[3][s]={'x': [],'y': []}
        data[4][s]={'x': [],'y': []}
        for image,target in loader:
            label=target.numpy()
            if label==0:
                data[0][s]['x'].append(image)
                data[0][s]['y'].append(0)
            if label==1:
                data[0][s]['x'].append(image)
                data[0][s]['y'].append(1)

            if label==2:
                data[1][s]['x'].append(image)
                data[1][s]['y'].append(0)
            if label==3:
                data[1][s]['x'].append(image)
                data[1][s]['y'].append(1)

            if label==4:
                data[2][s]['x'].append(image)
                data[2][s]['y'].append(0)
            if label==5:
                data[2][s]['x'].append(image)
                data[2][s]['y'].append(1)

            if label==6:
                data[3][s]['x'].append(image)
                data[3][s]['y'].append(0)
            if label==7:
                data[3][s]['x'].append(image)
                data[3][s]['y'].append(1)

            if label==8:
                data[4][s]['x'].append(image)
                data[4][s]['y'].append(0)
            if label==9:
                data[4][s]['x'].append(image)
                data[4][s]['y'].append(1)


    # "Unify" and save
    for n in range(5):
        for s in ['train','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'].clone()
        data[t]['valid']['y']=data[t]['train']['y'].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

########################################################################################################################
