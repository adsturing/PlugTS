import numpy as np
import argparse
import time
import torch
import os

from data import data
from PlugTS import PlugTS
from PlugTS_P import PlugTS_P

def exp_main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    b = data(args.dataset, is_shuffle=True, seed=2020)
    data_context = []
    data_reward = []
    if args.test:
        data_size = 11000
    else:
        data_size = 1000
    for t in range(min(data_size, b.size)):
        if args.test and t < 1000:
            continue
        context, rwd = b.step()
        data_context.append(context)
        data_reward.append(rwd)
    data_context = np.array(data_context)
    data_reward = np.array(data_reward)
    
    if args.model  == 'PlugTS':
        l = PlugTS(data_context.shape[-1], args.lamdba, args.nu, args.hidden)
    elif args.model  == 'PlugTS_P':
        l = PlugTS_P(data_context.shape[-1], args.lamdba, args.nu, args.hidden)
    else:
        raise ValueError('No such model')

    regrets = []
    res_regrets = []
    select_time = []
    all_idx = np.arange(data_context.shape[0])
    np.random.shuffle(all_idx)
    for t in range(len(all_idx)):
        context = data_context[all_idx[t]]
        rwd = data_reward[all_idx[t]]
        arm_select = l.select(context)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        loss = l.train(context[arm_select], r)
        torch.cuda.empty_cache()
        regrets.append(reg)
        res_regrets.append(np.sum(regrets))
        if t % 100 == 0:
            print('step {}  regret: {:.3f}'.format(t, np.sum(regrets)))
    print('step {}  regret: {:.3f}'.format(t, np.sum(regrets)))
    
if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id', '--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('-test', '--test', type=int, default=1, help='test or not')
    parser.add_argument('-dataset', '--dataset', default='mnist', help='dataset')
    parser.add_argument('-mo', '--model', type=str, default= 'PlugTS', help='Model name.')
    parser.add_argument('-nu', '--nu', type=float, default=0.001, help='nu for control variance')
    parser.add_argument('-lamdba', '--lamdba', type=float, default=0.01, help='lambda for regularzation')    
    parser.add_argument('-hidden', '--hidden', type=int, default=100, help='network hidden size')    
    args = parser.parse_args()
    exp_main(args)