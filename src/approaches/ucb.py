import os,sys,time
import numpy as np
import copy
import math
import torch
import torch.nn.functional as F
from .utils import BayesianSGD


class Appr(object):

    def __init__(self,model,args,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=1000):
        self.model=model
        self.device = args.device
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.init_lr=args.lr
        self.sbatch=args.sbatch
        self.nepochs=args.nepochs

        self.arch=args.arch
        self.samples=args.samples
        self.lambda_=1.

        self.output=args.output
        self.checkpoint = args.checkpoint
        self.experiment=args.experiment
        self.num_tasks=args.num_tasks

        self.modules_names_with_cls = self.find_modules_names(with_classifier=True)
        self.modules_names_without_cls = self.find_modules_names(with_classifier=False)



    def train(self,t,xtrain,ytrain,xvalid,yvalid):

        # Update the next learning rate for each parameter based on their uncertainty
        params_dict = self.update_lr(t)
        self.optimizer = BayesianSGD(params=params_dict)

        best_loss=np.inf

        # best_model=copy.deepcopy(self.model)
        best_model = copy.deepcopy(self.model.state_dict())
        lr = self.init_lr
        patience = self.lr_patience


        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                self.train_epoch(t,xtrain,ytrain)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain,ytrain)
                clock2=time.time()

                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),
                    train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

                if math.isnan(valid_loss) or math.isnan(train_loss):
                    print("saved best model and quit because loss became nan")
                    break

                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=copy.deepcopy(self.model.state_dict())
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience

                        params_dict = self.update_lr(t, adaptive_lr=True, lr=lr)
                        self.optimizer=BayesianSGD(params=params_dict)

                print()
        except KeyboardInterrupt:
            print()

        # Restore best
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.save_model(t)




    def update_lr(self,t, lr=None, adaptive_lr=False):
        params_dict = []
        if t==0:
            params_dict.append({'params': self.model.parameters(), 'lr': self.init_lr})
        else:
            for name in self.modules_names_without_cls:
                n = name.split('.')
                if len(n) == 1:
                    m = self.model._modules[n[0]]
                elif len(n) == 3:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
                elif len(n) == 4:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
                else:
                    print (name)

                if adaptive_lr is True:
                    params_dict.append({'params': m.weight_rho, 'lr': lr})
                    params_dict.append({'params': m.bias_rho, 'lr': lr})

                else:
                    w_unc = torch.log1p(torch.exp(m.weight_rho.data))
                    b_unc = torch.log1p(torch.exp(m.bias_rho.data))

                    params_dict.append({'params': m.weight_mu, 'lr': torch.mul(w_unc,self.init_lr)})
                    params_dict.append({'params': m.bias_mu, 'lr': torch.mul(b_unc,self.init_lr)})
                    params_dict.append({'params': m.weight_rho, 'lr':self.init_lr})
                    params_dict.append({'params': m.bias_rho, 'lr':self.init_lr})

        return params_dict


    def find_modules_names(self, with_classifier=False):
        modules_names = []
        for name, p in self.model.named_parameters():
            if with_classifier is False:
                if not name.startswith('classifier'):
                    n = name.split('.')[:-1]
                    modules_names.append('.'.join(n))
            else:
                n = name.split('.')[:-1]
                modules_names.append('.'.join(n))

        modules_names = set(modules_names)

        return modules_names

    def logs(self,t):

        lp, lvp = 0.0, 0.0
        for name in self.modules_names_without_cls:
            n = name.split('.')
            if len(n) == 1:
                m = self.model._modules[n[0]]
            elif len(n) == 3:
                m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
            elif len(n) == 4:
                m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]

            lp += m.log_prior
            lvp += m.log_variational_posterior

        lp += self.model.classifier[t].log_prior
        lvp += self.model.classifier[t].log_variational_posterior

        return lp, lvp


    def train_epoch(self,t,x,y):

        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).to(self.device)

        num_batches = len(x)//self.sbatch
        j=0
        # Loop batches
        for i in range(0,len(r),self.sbatch):

            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images, targets = x[b].to(self.device), y[b].to(self.device)

            # Forward
            loss=self.elbo_loss(images,targets,t,num_batches,sample=True).to(self.device)

            # Backward
            self.model.cuda()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.model.cuda()

            # Update parameters
            self.optimizer.step()
        return


    def eval(self,t,x,y,debug=False):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.as_tensor(r, device=self.device, dtype=torch.int64)

        with torch.no_grad():
            num_batches = len(x)//self.sbatch
            # Loop batches
            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images, targets = x[b].to(self.device), y[b].to(self.device)

                # Forward
                outputs=self.model(images,sample=False)
                output=outputs[t]
                loss = self.elbo_loss(images, targets, t, num_batches,sample=False,debug=debug)

                _,pred=output.max(1, keepdim=True)

                total_loss += loss.detach()*len(b)
                total_acc += pred.eq(targets.view_as(pred)).sum().item() 
                total_num += len(b)           

        return total_loss/total_num, total_acc/total_num


    def set_model_(model, state_dict):
        model.model.load_state_dict(copy.deepcopy(state_dict))


    def elbo_loss(self, input, target, t, num_batches, sample,debug=False):
        if sample:
            lps, lvps, predictions = [], [], []
            for i in range(self.samples):
                predictions.append(self.model(input,sample=sample)[t])
                lp, lv = self.logs(t)
                lps.append(lp)
                lvps.append(lv)

            # hack
            w1 = 1.e-3
            w2 = 1.e-3
            w3 = 5.e-2

            outputs = torch.stack(predictions,dim=0).to(self.device)
            log_var = w1*torch.as_tensor(lvps, device=self.device).mean()
            log_p = w2*torch.as_tensor(lps, device=self.device).mean()
            nll = w3*torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum').to(device=self.device)

            return (log_var - log_p)/num_batches + nll

        else:
            predictions = []
            for i in range(self.samples):
                pred = self.model(input,sample=False)[t]
                predictions.append(pred)


            # hack
            # w1 = 1.e-3
            # w2 = 1.e-3
            w3 = 5.e-6

            outputs = torch.stack(predictions,dim=0).to(self.device)
            nll = w3*torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum').to(device=self.device)

            return nll


        # w1, w2, w3 = self.get_coefs(nll,log_var,log_p,num_batches)
        # print ("New coefficients for task {} are w1={}, w2={}, w3={}".format(t,w1,w2,w3))
        # if math.isnan(log_var) or math.isnan(log_p) or math.isnan(nll):
        #     nll = torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum')
        # # if log_var > 1e3 or log_p > 1e3 or nll>1e3:
        #     print ("BEFORE: ", (log_var/num_batches).item(), (log_p / num_batches).item(), nll.item())
        #     # while math.isnan(nll):
        #         # nll = 1e-5*torch.nn.functional.nll_loss(outputs.mean(0), target, reduction='sum')


    def save_model(self,t):
        torch.save({'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(t)))



    # def get_coefs(self,nll,log_var,log_p,num_batches):
    #     def take_n(num):
    #         return torch.log10(num).item()
    #
    #     exponents = np.array([take_n(num) for num in [nll, log_p, log_var]])
    #     min_exp = exponents.min()
    #     min_exp_idx = np.argmin(exponents)
    #     if min_exp_idx == 0:
    #         w1 = (10**(3-(take_n(log_var)+min_exp)))*num_batches
    #         w2 = (10**-(3-(take_n(log_p)+min_exp)))*num_batches
    #         w3 = 10.**(3-min_exp_idx)
    #     if min_exp_idx == 1:
    #         w1 = (10**(3-(take_n(log_var)+min_exp)))*num_batches
    #         w3 = 10**(3-(take_n(nll)+min_exp))
    #         w2 = (10.**-(3-min_exp_idx))*num_batches
    #     if min_exp_idx == 2:
    #         w3 = 10**(3-(take_n(nll)+min_exp))
    #         w2 = (10**-(3-(take_n(log_p)+min_exp)))*num_batches
    #         w1 = (10.**(3-min_exp_idx))*num_batches
    #
    #     return w1, w2, w3


