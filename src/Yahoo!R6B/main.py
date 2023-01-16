# -*- coding: utf-8 -*-
import numpy as np
import fileinput
import torch
from torch.multiprocessing import Pool 
import argparse
import os

from PlugTS_P import PlugTS_P
from PlugTS import PlugTS

def process(idx_user):
    head = item_candidates_length[idx_user]
    tail = item_candidates_length[idx_user+1]
    ctr_final = ctr_pred[head:tail]
    if item_candidates[idx_user][ctr_final.argmax()] == x_item_r6[idx_user]:
        return 1
    else:
        return 0

def exp_main (config):    
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    global  item_candidates_length, ctr_pred, item_candidates,x_item_r6
    all_item = {}
    exposure_item_cumu = np.zeros(653)
    train_time = []
    data_dir = config.dataset + '/ydata-fp-td-clicks-v2_0.201110'
    if config.test:
        names = ["03","04","05","06","07","08","09","10","11","12","13","14","15","16"]
    else:
        names = ["02"]
    
    num_selected = {}
    ctr_selected = {}
    sw_selected = {}
    mean_num_selected = []
    sum_num_selected = []
    result_ctr = []
    result_sw = []
    times_update = 0
    min_len_candidates = 1e10
    max_len_candidates = 0

    for name in names:   
        num_selected[name] = []
        sw_selected[name] = []
        ctr_selected[name] = []
        x_user_r6 = []
        x_item_r6 = []
        y_r6 = []
        item_candidates = []
        item_candidates_all = []
        item_candidates_length = [0]
        user_candidates_all = []
        
        files = data_dir + name
        with fileinput.input(files) as f:
            for line in f:
                cols = line.split()
                if int(cols[1][3:]) in all_item:
                    pass
                else:
                    all_item[int(cols[1][3:]) ] = len(all_item)+1
                x_item_r6.append(all_item[int(cols[1][3:])])
                y_r6.append(int(cols[2]))
                x_user_r6.append([0]*136)
                i = 4
                while not cols[i][0] == '|':
                    x_user_r6[-1][int(cols[i])-1] = 1
                    i += 1
                item_candidates_length.append(item_candidates_length[-1])
                item_candidates.append([])
                while i < len(cols):
                    if int(cols[i][4:]) in all_item:
                        pass
                    else:
                        all_item[int(cols[i][4:])] = len(all_item)+1
                    item_candidates[-1].append(all_item[int(cols[i][4:])])
                    item_candidates_all.append(all_item[int(cols[i][4:])])
                    item_candidates_length[-1] += 1
                    user_candidates_all.append(x_user_r6[-1])
                    i += 1

                if len(y_r6) == 8e4:                    
                    print('round',times_update+1,": simulate with data in day",name)
                    data_selected = []
                    if times_update > 0:
                        len_candidates = len(user_candidates_all)
                        min_len_candidates = min(min_len_candidates, len_candidates)
                        max_len_candidates = max(max_len_candidates, len_candidates)
                        gap = 100000
                        for cand_idx in range(0, len_candidates, gap):
                            if cand_idx ==0:
                                ctr_pred = model.predict(user_candidates_all[cand_idx:min(cand_idx+gap,len_candidates)] , 
                                                    item_candidates_all[cand_idx:min(cand_idx+gap,len_candidates)])
                            else:
                                ctr_pred = np.concatenate((ctr_pred, 
                                                            model.predict(user_candidates_all[cand_idx:min(cand_idx+gap,len_candidates)] , 
                                                                        item_candidates_all[cand_idx:min(cand_idx+gap,len_candidates)])))

                        pool = Pool()
                        data_selected = pool.map(process, range(len(x_user_r6)))
                        pool.close()
                        pool.join()
                        data_selected = np.array(data_selected).astype(int)

                    x_user_r6 = np.array(x_user_r6).astype(int)
                    x_item_r6 = np.array(x_item_r6).astype(int)
                    y_r6 = np.array(y_r6).astype(int)

                    if len(data_selected)>0:
                        x_user_r6 = x_user_r6[data_selected==1]
                        x_item_r6 = x_item_r6[data_selected==1]
                        y_r6 = y_r6[data_selected==1]

                        num_selected[name].append(x_user_r6.shape[0])
                        sw_selected[name].append(y_r6[y_r6==1].shape[0])
                        ctr_selected[name].append(y_r6[y_r6==1].shape[0]/y_r6.shape[0])                        
                        for item_exp in x_item_r6:
                            exposure_item_cumu[item_exp] += 1

                        train_x_user_new = torch.Tensor(x_user_r6).cuda()
                        train_x_user = torch.cat((train_x_user[train_x_user_new.size(0):], train_x_user_new),dim=0)
                        train_x_item_new = torch.LongTensor(x_item_r6).cuda()
                        train_x_item = torch.cat((train_x_item[train_x_item_new.size(0):], train_x_item_new),dim=0)
                        train_y_new = torch.Tensor(y_r6).cuda()
                        train_y = torch.cat((train_y[train_y_new.size()[0]:], train_y_new),dim=0)
                    else:
                        train_x_user = torch.Tensor(x_user_r6).cuda()
                        train_x_item = torch.LongTensor(x_item_r6).cuda()
                        train_y = torch.Tensor(y_r6).cuda()

                    exposure_item_last = np.zeros(653)
                    min_item = 500
                    max_item = 0
                    for item_exp in train_x_item:
                        exposure_item_last[item_exp] += 1
                        min_item = min(min_item, item_exp)
                        max_item = max(max_item, item_exp)

                    if config.model == 'PlugTS':
                        model = PlugTS(num_users=136, num_items=653, embedding_k=config.embedding_dimention, nu=config.nu)
                    elif config.model == 'PlugTS_P':
                        model = PlugTS_P(num_users=136, num_items=653, embedding_k=config.embedding_dimention, nu=config.nu)
                    else:
                        raise ValueError('No such model')
                        
                    model.fit(train_x_user, train_x_item, train_y, 
                                num_epoch = config.epoch,
                                lr = config.learningRate, 
                                batch_size = config.batchSize, 
                                lamb = config.weight_decay_lamb)

                    times_update += 1
                    x_user_r6 = []
                    x_item_r6 = []
                    y_r6 = []
                    item_candidates = []
                    item_candidates_all = []
                    item_candidates_length = [0]
                    user_candidates_all = []

        ctr_now = np.array(ctr_selected[name])
        sw_now = np.array(sw_selected[name])
        result_ctr.append(float(ctr_now.mean()))
        result_sw.append(int(sw_now.sum()))

    print("ctr:",result_ctr)
    print("sw:",result_sw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu_id', '--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('-dataset', '--dataset', type=str, default= "./data/Yahoo!R6B", help='dataset')
    parser.add_argument('-mo', '--model', type=str, default= 'PlugTS', help='Model name.')
    parser.add_argument('-test', '--test', type=int, default=1, help='test or not')

    parser.add_argument('-ep', '--epoch', type=int,default=1, help='epoch number.')
    parser.add_argument('-embed_dim', '--embedding_dimention', type=int, default=6, help='embedding_dimention')
    parser.add_argument('-lr', '--learningRate', type=float, default=1e-2, help='learningRate.')
    parser.add_argument('-bs', '--batchSize', type=int, default=64, help='batch size')

    parser.add_argument('-lamb', '--weight_decay_lamb', type=float, default=0, help='weight_decay.')
    parser.add_argument('-nu', '--nu', type=float, default=1, help='nu for control variance')

    config, _ = parser.parse_known_args ()
    exp_main(config)

    