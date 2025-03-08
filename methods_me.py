import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from optimizers import get_optimizer
from backbones import networks
from FDA_1d import FDA_1d_with_fs
import torch.nn.functional as F


class Algorithm(torch.nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        # if name is None:
        #     name=self.hparams["optimizer"]
        optimizer = get_optimizer(self.hparams,parameters)
            
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone

class Baseline(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Baseline, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        print("Building F")
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # gesture network
        self.classifier_g = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.classifier_g = networks.MLP(self.featurizer.n_outputs, num_classes,self.hparams)
        # style network
        # self.classifier_s = nn.Linear(self.featurizer.n_outputs, num_domains)
        # self.classifier_s = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        
        
        self.optimizer_f = self.new_optimizer(self.featurizer.parameters())
        self.optimizer_g = self.new_optimizer(self.classifier_g.parameters())
        # self.optimizer_s = self.new_optimizer(self.classifier_s.parameters())
        # self.weight_adv = hparams["w_adv"]

    def forward_g(self, x):
        # learning gesture network on randomized style
        # return self.classifier_g(self.randomize(self.featurizer(x), "style"))
        return self.classifier_g(self.featurizer(x))


    def update(self, minibatches,all_x_test,args,unlabeled=None):
        # all_x = torch.cat([x for x, y in minibatches])
        all_x=minibatches[0]
        # all_y = torch.cat([y for x, y in minibatches])
        all_y=minibatches[1]
        

        if args.high==1000:
            args.high=None
        
        all_x_src_to_tar=FDA_1d_with_fs(all_x,all_x_test,fs=1000,cutoff_freq=args.low,cutoff_freq_upper=args.high)
        # s_t_a1,s_t_a2,s_t_a3=self.decomse(all_x_src_to_tar)
        # all_d = torch.cat([
        #     torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
        #     for i, (x, y) in enumerate(minibatches)
        # ])

        # learn gesture feature
        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        # print(self.featurizer(all_x_src_to_tar).shape)
        b,a,c,T=all_x_src_to_tar.shape
        all_x_src_to_tar=all_x_src_to_tar.reshape(b,a*c,T)
        loss_g = F.cross_entropy(self.forward_g(all_x_src_to_tar), all_y)
        loss_g.backward()
        self.optimizer_f.step()
        self.optimizer_g.step()  

        return {
            "loss_g": loss_g.item(),
        }

    def predict(self, x):
        # x_a1,x_a2,x_a3=self.decomse(x)
        b,a,c,T=x.shape
        x=x.reshape(b,a*c,T)
        return self.classifier_g(self.featurizer(x))  
    
    def decomse(self,x):

        # b,a,c,t=x.shape

        x_a1,x_a2,x_a3=x[:,0,:,:],x[:,1,:,:],x[:,2:,:,:]
        # print(x_a1.shape)
        # exit(0)
        return x_a1,x_a2,x_a3
    


class Baseline_two_attn(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Baseline_two_attn, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        print("Building F")
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # gesture network
        self.classifier_g = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.classifier_g = networks.MLP(self.featurizer.n_outputs, num_classes,self.hparams)
        # style network
        # self.classifier_s = nn.Linear(self.featurizer.n_outputs, num_domains)
        # self.classifier_s = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        
        
        self.optimizer_f = self.new_optimizer(self.featurizer.parameters())
        self.optimizer_g = self.new_optimizer(self.classifier_g.parameters())
        # self.optimizer_s = self.new_optimizer(self.classifier_s.parameters())
        # self.weight_adv = hparams["w_adv"]

    def forward_g(self, x):
        # learning gesture network on randomized style
        # return self.classifier_g(self.randomize(self.featurizer(x), "style"))
        return self.classifier_g(self.featurizer(x))
    
    def forward_f(self,x):
        return self.featurizer(x)
    
    def forward_c(self,x):
        return self.classifier_g(x)


    def update(self, minibatches,all_x_test,args,unlabeled=None):
        # all_x = torch.cat([x for x, y in minibatches])
        all_x=minibatches[0]
        # all_y = torch.cat([y for x, y in minibatches])
        all_y=minibatches[1]
        

        if args.high==1000:
            args.high=None
        
        # all_x_src_to_tar=FDA_1d_with_fs(all_x,all_x_test,fs=1000,cutoff_freq=args.low,cutoff_freq_upper=args.high)
        s_t_a1,s_t_a2,s_t_a3=self.decomse(all_x)
        # s_t_a1,s_t_a2,s_t_a3=self.decomse(all_x_src_to_tar)
        # all_d = torch.cat([
        #     torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
        #     for i, (x, y) in enumerate(minibatches)
        # ])

        # learn gesture feature
        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        # print(self.featurizer(all_x_src_to_tar).shape)
        f_s_t_a1=self.forward_f(s_t_a1)
        f_s_t_a2=self.forward_f(s_t_a2)
        
        loss_g = F.cross_entropy(self.forward_c(f_s_t_a1+f_s_t_a2), all_y)
        loss_g.backward()
        self.optimizer_f.step()
        self.optimizer_g.step()  

        return {
            "loss_g": loss_g.item(),
        }

    def predict(self, x):
        
        x_a1,x_a2,x_a3=self.decomse(x)
        return self.classifier_g(self.forward_f(x_a1)+self.forward_f(x_a2))  
    def decomse(self,x):


        x_a1,x_a2,x_a3=x[:,0,:,:],x[:,1,:,:],x[:,2:,:,:]
        # print(x_a1.shape)
        # exit(0)
        return x_a1,x_a2,x_a3
    



class Baseline_with_arc(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Baseline_with_arc, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        print("Building F")
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # gesture network
        self.classifier_g = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.classifier_g = networks.MLP(self.featurizer.n_outputs, num_classes,self.hparams)
        # style network
        # self.classifier_s = nn.Linear(self.featurizer.n_outputs, num_domains)
        # self.classifier_s = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        
        
        self.optimizer_f = self.new_optimizer(self.featurizer.parameters())
        self.optimizer_g = self.new_optimizer(self.classifier_g.parameters())
        # self.optimizer_s = self.new_optimizer(self.classifier_s.parameters())
        # self.weight_adv = hparams["w_adv"]

    def forward_g(self, x):
        # learning gesture network on randomized style
        # return self.classifier_g(self.randomize(self.featurizer(x), "style"))
        return self.classifier_g(self.featurizer(x))
    
    def forward_f(self,x):
        return self.featurizer(x)
    
    def forward_c(self,x):
        return self.classifier_g(x)



    def update(self, minibatches,all_x_test,args,unlabeled=None):
        # all_x = torch.cat([x for x, y in minibatches])
        all_x=minibatches[0]
        # all_y = torch.cat([y for x, y in minibatches])
        all_y=minibatches[1]
        

        if args.high==1000:
            args.high=None
        
        all_x_src_to_tar=FDA_1d_with_fs(all_x,all_x_test,fs=1000,cutoff_freq=args.low,cutoff_freq_upper=args.high)
        s_t_a1,s_t_a2,s_t_a3=self.decomse(all_x_src_to_tar)
        # s_t_a1,s_t_a2,s_t_a3=self.decomse(all_x)
        # print(s_t_a1.shape)

        # s_t_a1,s_t_a2,s_t_a3=self.decomse(all_x_src_to_tar)
        # all_d = torch.cat([
        #     torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
        #     for i, (x, y) in enumerate(minibatches)
        # ])

        # learn gesture feature
        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        # print(self.featurizer(all_x_src_to_tar).shape)

        s_t_f_a1,s_t_f_a2=self.forward_f(s_t_a1),self.forward_f(s_t_a2)
        # loss_arc=self.cosine_similarity_loss(s_t_f_a1,s_t_f_a2)
        loss_arc=self.cosine_similarity_loss_cos(s_t_f_a1,s_t_f_a2)
        loss_g = F.cross_entropy(self.forward_c(s_t_f_a1+s_t_f_a2), all_y)
        total_loss = loss_arc + loss_g
        total_loss.backward()
        
        self.optimizer_f.step()
        self.optimizer_g.step()  

        return {
            "loss_g": loss_g.item(),
            "loss_arc":loss_arc.item()
        }

    def predict(self, x):
        x_a1,x_a2,x_a3=self.decomse(x)
        return self.classifier_g(self.featurizer(x_a1)+self.featurizer(x_a2))  
    
    def decomse(self,x):

        # b,a,c,t=x.shape

        x_a1,x_a2,x_a3=x[:,0,:,:],x[:,1,:,:],x[:,2:,:,:]
        # print(x_a1.shape)
        # exit(0)
        return x_a1,x_a2,x_a3
        
    def cosine_similarity_loss(self, feature1, feature2):
        """
        计算两个特征向量之间的余弦相似度损失。
        
        参数:
            feature1 (torch.Tensor): 第一个特征向量，形状为 (batch_size, feature_dim)。
            feature2 (torch.Tensor): 第二个特征向量，形状为 (batch_size, feature_dim)。
        
        返回:
            loss (torch.Tensor): 余弦相似度损失，值越小表示特征越相似。
        """
        # 归一化特征向量
        feature1_normalized = F.normalize(feature1, p=2, dim=1)
        feature2_normalized = F.normalize(feature2, p=2, dim=1)
        
        #todo 
        #cos + 符号
        # 计算余弦相似度
        cosine_sim = torch.sum(feature1_normalized * feature2_normalized, dim=1)
        
        # 计算损失（1 - 余弦相似度）
        loss =- cosine_sim.mean()
        
        return loss
    
    def cosine_similarity_loss_cos(self,feature1,feature2):

        loss_arc=-F.cosine_similarity(feature1,feature2).sum()/feature1.shape[0]
        return loss_arc
    

class Baseline_with_arc_concat(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Baseline_with_arc_concat, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        print("Building F")
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # gesture network
        self.classifier_g = nn.Linear(self.featurizer.n_outputs *2, num_classes)
        # self.classifier_g = networks.MLP(self.featurizer.n_outputs, num_classes,self.hparams)
        # style network
        # self.classifier_s = nn.Linear(self.featurizer.n_outputs, num_domains)
        # self.classifier_s = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        
        
        self.optimizer_f = self.new_optimizer(self.featurizer.parameters())
        self.optimizer_g = self.new_optimizer(self.classifier_g.parameters())
        # self.optimizer_s = self.new_optimizer(self.classifier_s.parameters())
        # self.weight_adv = hparams["w_adv"]

    def forward_g(self, x):
        # learning gesture network on randomized style
        # return self.classifier_g(self.randomize(self.featurizer(x), "style"))
        return self.classifier_g(self.featurizer(x))
    
    def forward_f(self,x):
        return self.featurizer(x)
    
    def forward_c(self,x):
        return self.classifier_g(x)



    def update(self, minibatches,all_x_test,args,unlabeled=None):
        # all_x = torch.cat([x for x, y in minibatches])
        all_x=minibatches[0]
        # all_y = torch.cat([y for x, y in minibatches])
        all_y=minibatches[1]
        

        if args.high==1000:
            args.high=None
        
        all_x_src_to_tar=FDA_1d_with_fs(all_x,all_x_test,fs=1000,cutoff_freq=args.low,cutoff_freq_upper=args.high)
        s_t_a1,s_t_a2,s_t_a3=self.decomse(all_x_src_to_tar)
        # s_t_a1,s_t_a2,s_t_a3=self.decomse(all_x)
        # print(s_t_a1.shape)

        # s_t_a1,s_t_a2,s_t_a3=self.decomse(all_x_src_to_tar)
        # all_d = torch.cat([
        #     torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
        #     for i, (x, y) in enumerate(minibatches)
        # ])

        # learn gesture feature
        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        # print(self.featurizer(all_x_src_to_tar).shape)

        s_t_f_a1,s_t_f_a2=self.forward_f(s_t_a1),self.forward_f(s_t_a2)
        # loss_arc=self.cosine_similarity_loss(s_t_f_a1,s_t_f_a2)
        loss_arc=self.cosine_similarity_loss_cos(s_t_f_a1,s_t_f_a2)
        loss_g = F.cross_entropy(self.forward_c(torch.cat((s_t_f_a1,s_t_f_a2),dim=-1)), all_y)
        total_loss = loss_arc + loss_g
        total_loss.backward()
        
        self.optimizer_f.step()
        self.optimizer_g.step()  

        return {
            "loss_g": loss_g.item(),
            "loss_arc":loss_arc.item()
        }

    def predict(self, x):
        x_a1,x_a2,x_a3=self.decomse(x)
        return self.classifier_g(torch.cat((self.featurizer(x_a1),self.featurizer(x_a2)),dim=-1))  
    
    def decomse(self,x):

        # b,a,c,t=x.shape

        x_a1,x_a2,x_a3=x[:,0,:,:],x[:,1,:,:],x[:,2:,:,:]
        # print(x_a1.shape)
        # exit(0)
        return x_a1,x_a2,x_a3
        
    def cosine_similarity_loss(self, feature1, feature2):
        """
        计算两个特征向量之间的余弦相似度损失。
        
        参数:
            feature1 (torch.Tensor): 第一个特征向量，形状为 (batch_size, feature_dim)。
            feature2 (torch.Tensor): 第二个特征向量，形状为 (batch_size, feature_dim)。
        
        返回:
            loss (torch.Tensor): 余弦相似度损失，值越小表示特征越相似。
        """
        # 归一化特征向量
        feature1_normalized = F.normalize(feature1, p=2, dim=1)
        feature2_normalized = F.normalize(feature2, p=2, dim=1)
        
        #todo 
        #cos + 符号
        # 计算余弦相似度
        cosine_sim = torch.sum(feature1_normalized * feature2_normalized, dim=1)
        
        # 计算损失（1 - 余弦相似度）
        loss =- cosine_sim.mean()
        
        return loss
    
    def cosine_similarity_loss_cos(self,feature1,feature2):

        loss_arc=-F.cosine_similarity(feature1,feature2).sum()/feature1.shape[0]
        return loss_arc