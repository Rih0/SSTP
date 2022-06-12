import argparse
import torch
import torch.nn as nn
import numpy as np
from utils import Sampler, evaluate, load_dataset, poi_category_relation, geographical_info_build, user_social_network
from model import SSTP
import os
import random
import time

parser = argparse.ArgumentParser(description="the code of SSTP.")
parser.add_argument("--dataset", type=str, default="NYC", help="the input dataset.")
parser.add_argument("--device", type=str, default="cuda", help="use GPU('cuda') or CPU('cpu').")
parser.add_argument("--gpu", type=int, default=0, help="which GPU you want to use. use CPU please setting -1.")
parser.add_argument("--epochs", type=int, default=51, help="number of training epochs.")
parser.add_argument("--lr", type=float, default=0.001, help="the learning rate of model.")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay.")
parser.add_argument("--feat-drop", type=float, default=0.6, help="input features dropout.")
parser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout.")
parser.add_argument("--negative-slop", type=float, default=0.2, help="the negative slope of leaky relu. same to alpha.")
parser.add_argument("--fastmode", action="store_true", default=False, help="skip re-evaluate the vaildation set.")
parser.add_argument("--hidden", type=int, default=8, help="the number of  hidden units size of model.")
parser.add_argument("--layers", type=int, default=1, help="the number of hidden layers of model.")
parser.add_argument("--num-heads", type=int, default=1, help="number of hidden graph attention network heads.")
parser.add_argument("--num-out-heads", type=int, default=1, help="number of output attention heads.")
parser.add_argument("--seed", type=int, default=2022, help="random seed.")
parser.add_argument("--batch-size", type=int, default=128, help="the size of one batch.")
parser.add_argument("--negative-sample-num", type=int, default=10, help="number of negative sample.")
parser.add_argument("--model", type=str, default="SSTP", help="recommendation model name.")
parser.add_argument("--topks", nargs='?', default="[20]", help="@k test list. such as: [1, 5, 10, 20].")
parser.add_argument("--hidden-size", type=int, default=100, help="the hidden size of user, poi, time, category embedding.")

parser.add_argument("--user-hidden-size", type=int, default=100, help="the hidden size of user embedding.")
parser.add_argument("--poi-hidden-size", type=int, default=100, help="the hidden size of poi embedding.")
parser.add_argument("--time-hidden-size", type=int, default=100, help="the hidden size of time embedding.")
parser.add_argument("--cate-hidden-size", type=int, default=100, help="the hidden size of category embedding.")

parser.add_argument("--time-slot", type=int, default=24, help="time slot.")
parser.add_argument("--max-len", type=int, default=50, help="max len of user checkin sequence.")
parser.add_argument("--dropout", type=float, default=0.5, help="the dropout of feat embedding.")
parser.add_argument("--num-layers", type=int, default=2, help="the number of attention layers of model.")
parser.add_argument("--rnn-num-layers", type=int, default=2, help="the number of GRU layers of model.") # 2
parser.add_argument('--l2-emb', default=0.2, type=float, help="L2.")
parser.add_argument('--limit', default=3, type=int, help="user distance of poi.")
parser.add_argument('--sim', default=0.4, type=float, help="the similarity between users.")
parser.add_argument("--cate-lambda", default=0.3, type=float, help="the rate of category criterion.")
parser.add_argument("--poi-lambda", default=0.5, type=float, help="the rate of poi criterion.")
parser.add_argument("--rnn-type", type=str, default="GRU", help="the type of rnn.")

args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Main funciton
if __name__ == "__main__":
    # 设置随机种子
    set_seed(args.seed)

    # 选择运行的CUDA驱动
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # 加载数据集
    dataset = load_dataset(args.dataset)
    [train_poi, valid_poi, test_poi,
     train_time, valid_time, test_time,
     train_cate, valid_cate, test_cate,
     user_num, poi_num, cate_num] = dataset
    print("### user number : {:5d} ###".format(user_num))
    print("### poi  number : {:5d} ###".format(poi_num))
    print("### cate number : {:5d} ###".format(cate_num))
    print("args : {}".format(args))

    # 构建对应数据
    # poi_adj表示两个POI之间是否存在联系，poi_neighbors表示某个POI的所有邻居POI，poi_attention_coefficient表示POI与其邻居的距离相似度。=
    poi_adj, poi_neighbors, poi_attention_coefficient = geographical_info_build(poi_num, args.dataset, limit=args.limit)    #.to(args.device)
    # POI与对应类别的对应关系
    poi2cate = poi_category_relation(args.dataset)
    # user_adj表示用户邻接图，user_poi_dict为字典形式：user_poi_dict[uid] = list(poi)，表示某一个用户的所有签到记录
    user_adj, user_poi_dict = user_social_network(args.dataset, args.sim)
    
    # Loading model
    model = SSTP(args, user_num, poi_num, args.time_slot, cate_num, poi_adj=poi_adj, poi2cate=poi2cate, 
                user_adj=user_adj, user_poi_dict=user_poi_dict, poi_attention_coefficient=poi_attention_coefficient,
                poi_neighbors=poi_neighbors)
    model = model.to(args.device)

    # dataset sampler
    train_sampler = Sampler(train_poi, train_time, train_cate, user_num, poi_num, cate_num, batch_size=args.batch_size, max_len=args.max_len, n_workers=3)
    # init parameters
    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)    # only 2-dim above can be init.
        except:
            pass

    # criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    cross_entropy_loss = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    
    # the number of batch size
    num_batch = len(train_poi) // args.batch_size
    begin_time = time.time()
    
    # Training
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step in range(num_batch):
            # 采样器取出数据
            u, poi, tim, cate, poi_y, cat_y = train_sampler.next_batch()
            u, poi, tim, cate, poi_y, cat_y = np.array(u), np.array(poi), np.array(tim), np.array(cate), np.array(poi_y), np.array(cat_y)
            
            # 模型输出
            pred_poi, pred_cate = model(u, poi, tim, cate)
            poi_y, cate_y = torch.LongTensor(poi_y).to(args.device), torch.LongTensor(cat_y).to(args.device)

            # 损失计算
            loss_poi, loss_cate = 0, 0
            for i in range(u.shape[0]):
                loss_poi += cross_entropy_loss(pred_poi[i, :, :], poi_y[i, :])
            for i in range(u.shape[0]):
                loss_cate += cross_entropy_loss(pred_cate[i, :, :], cate_y[i, :])
            
            loss = loss_poi + args.cate_lambda * loss_cate
            # for param in model.poi_embedding.parameters(): loss += args.l2_emb * torch.norm(param)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                print("Evaluating.")
                # 加快程序运行速度，只选择 test_num 数量的测试数据进行测试
                if args.dataset == "NYC":
                    test_num = 183
                else:
                    test_num = 193
                HIT, MAP = evaluate(args, model, dataset, test_num)
                print("Epoch:{:5d}, [TEST] HIT@1:{}, HIT@5:{}, HIT@10:{}, HIT@20:{}, MAP@1:{}, MAP@5:{}, MAP@10:{}, MAP@20:{}".format(
                    epoch, HIT[0], HIT[1], HIT[2], HIT[3],
                    MAP[0], MAP[1], MAP[2], MAP[3]))

    # Done
    print("Finish time : {}s.".format(time.time() - begin_time))
