import os
import numpy as np
import gzip
import pickle
from copy import deepcopy

########################################################################################################################
def print_arguments(args):
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)


def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())

    print()
    print('Num parameters = %s'%(human_format(count)), human_format(sum(p.numel() for p in model.parameters())))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################


def save_log(taskcla, acc, lss, data, output_path):
    logs = {}
    # save task names
    logs['task_name'] = {}
    logs['test_acc'] = {}
    logs['test_loss'] = {}
    for t, ncla in taskcla:
        logs['task_name'][t] = deepcopy(data[t]['name'])
        logs['test_acc'][t] = deepcopy(acc[t, :])
        logs['test_loss'][t] = deepcopy(lss[t, :])
    # pickle
    with gzip.open(os.path.join(output_path, 'logs.p'), 'wb') as output:
        pickle.dump(logs, output, pickle.HIGHEST_PROTOCOL)

    print ("Log file saved in ", os.path.join(output_path, 'logs.p'))

def make_directories(args):
    if args.output=='':
        args.output = '{}_{}'.format(args.experiment,args.approach)
        print (args.output)
    checkpoint = os.path.join(args.checkpoint_dir, args.output)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(checkpoint):os.mkdir(checkpoint)
    print("Results will be saved in ", checkpoint)

    return checkpoint


def print_log_acc_bwt(args, acc, lss):

    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end=',')
        for j in range(acc.shape[1]):
            print('{:5.4f}% '.format(acc[i,j]),end=',')
        print()

    avg_acc = np.mean(acc[acc.shape[0]-1,:])
    print ('ACC: {:5.4f}%'.format(avg_acc))
    print()
    print()

    ucb_bwt = (acc[-1] - np.diag(acc)).mean()
    print ('BWT : {:5.2f}%'.format(ucb_bwt))

    print('*'*100)
    print('Done!')

    logs = {}
    # save results
    logs['name'] = args.experiment
    logs['taskcla'] = args.taskcla
    logs['acc'] = acc
    logs['loss'] = lss
    logs['bwt'] = ucb_bwt
    logs['rii'] = np.diag(acc)
    logs['rij'] = acc[-1]

    # pickle
    path = os.path.join(args.checkpoint, '{}_{}_seed_{}.p'.format(args.experiment,args.approach, args.seed))
    with open(path, 'wb') as output:
        pickle.dump(logs, output)

    print ("Log file saved in ", path)
    return avg_acc, ucb_bwt

