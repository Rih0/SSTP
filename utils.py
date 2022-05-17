from multiprocessing import Process, Queue
from collections import defaultdict
from os import times
import numpy as np
import copy
import torch
import time
from math import radians, sin, cos, asin, sqrt
import dgl
import pickle

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(train_poi, train_time, train_cate, usernum, poinum, catenum, batch_size, max_len, result_queue, SEED):

    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(train_poi[user]) <= 1: 
            user = np.random.randint(1, usernum + 1)
        poi = np.zeros([max_len], dtype=np.int32)
        time = np.zeros([max_len], dtype=np.int32)
        cate = np.zeros([max_len], dtype=np.int32)

        poi_y = np.zeros([max_len], dtype=np.int32)
        cate_y = np.zeros([max_len], dtype=np.int32)

        nxt = train_poi[user][-1]                   
        idx = max_len - 1                           
        poi_ts = set(train_poi[user])
        for i in reversed(train_poi[user][:-1]):
            poi[idx] = i        
            poi_y[idx] = nxt - 1
            nxt = i             
            idx -= 1            
            if idx == -1: 
                break
        
        idx = max_len - 1
        for i in reversed(train_time[user][:-1]):
            time[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        nxt = train_cate[user][-1]
        idx = max_len - 1
        for i in reversed(train_cate[user][:-1]):
            cate[idx] = i
            cate_y[idx] = nxt - 1
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return (user, poi, time, cate, poi_y, cate_y)

    np.random.seed(SEED)    # 设置种子
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))

class Sampler(object):
    def __init__(self, Poi, Time, Cate, usernum, poinum, catenum, batch_size=64, max_len=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(Poi,
                                                      Time,
                                                      Cate,
                                                      usernum,
                                                      poinum,
                                                      catenum,
                                                      batch_size,
                                                      max_len,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def TZTime2Timestamp(s):
    return int(time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))

def load_dataset(dataset="NYC"):
    if dataset == "NYC" or dataset == "TKY":
        data_file_path = "./data/" + dataset + "/3_sort_total_u10p10_" + dataset + ".txt"
    else:
        None
    
    user_num, poi_num, cate_num = 0, 0, 0
    poi_checkin, time_checkin, cate_checkin = defaultdict(list), defaultdict(list), defaultdict(list)
    train_poi, valid_poi, test_poi = {}, {}, {}
    train_time, valid_time, test_time = {}, {}, {}
    train_cate, valid_cate, test_cate = {}, {}, {}

    with open(data_file_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            uid, pid, lat, lon, timestamp, did, cid, utc = line.strip().split("\t")
            uid, pid, timestamp, cid = int(uid), int(pid), int(timestamp), int(cid)
            user_num = max(uid, user_num)
            poi_num = max(pid, poi_num)
            cate_num = max(cid, cate_num)
            
            poi_checkin[uid].append(pid)
            time_checkin[uid].append(timestamp)
            cate_checkin[uid].append(cid)

    for u in range(1, user_num + 1):
        train_poi[u] = poi_checkin[u][:-2]
        valid_poi[u] = [poi_checkin[u][-2]]
        test_poi[u] = [poi_checkin[u][-1]]

        train_time[u] = time_checkin[u][:-2]
        valid_time[u] = [time_checkin[u][-2]]
        test_time[u] = [time_checkin[u][-1]]

        train_cate[u] = cate_checkin[u][:-2]
        valid_cate[u] = [cate_checkin[u][-2]]
        test_cate[u] = [cate_checkin[u][-1]] 
    
    return [train_poi, valid_poi, test_poi, \
            train_time, valid_time, test_time, \
            train_cate, valid_cate, test_cate, \
            user_num, poi_num, cate_num]

def evaluate(args, model, dataset, test_num):
    [train_poi, valid_poi, test_poi, \
        train_time, valid_time, test_time,
        train_cate, valid_cate, test_cate,
        user_num, poi_num, cate_num] = copy.deepcopy(dataset)
    
    users_list = range(1, user_num + 1)

    test_user_cnt = 0.0
    HIT, MAP = [0, 0, 0, 0], [0, 0, 0, 0]

    for u in users_list:
        poi_seqs = np.zeros([args.max_len], dtype=np.int32)
        tim_seqs = np.zeros([args.max_len], dtype=np.int32)
        cat_seqs = np.zeros([args.max_len], dtype=np.int32)

        idx = args.max_len - 1
        poi_seqs[idx] = valid_poi[u][0]
        idx -= 1
        for i in reversed(train_poi[u]):
            poi_seqs[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        idx = args.max_len - 1
        tim_seqs[idx] = valid_time[u][0]
        idx -= 1
        for i in reversed(train_time[u]):
            tim_seqs[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        idx = args.max_len - 1
        cat_seqs[idx] = valid_cate[u][0]
        idx -= 1
        for i in reversed(train_cate[u]):
            cat_seqs[idx] = i
            idx -= 1
            if idx == -1:
                break
    

        poi_scores, cate_scores = model(*[np.array(l) for l in [[u], [poi_seqs], [tim_seqs], [cat_seqs]]])
        poi_scores = poi_scores[:, -1, :]
        poi_id = test_poi[u][0] - 1 # true test label

        # compute hit
        hits, maps = [0, 0, 0, 0], [0, 0, 0, 0]
        for i, N in enumerate([1, 5, 10, 20]):
            _, topk_idx = torch.topk(poi_scores, k=N)
            for _, idx in enumerate(topk_idx):
                if poi_id in idx.tolist():
                    hits[i] += 1
                    maps[i] += (1 / (idx.tolist().index(poi_id) + 1))

        
        for i, N in enumerate([1, 5, 10, 20]):
            HIT[i] += hits[i]
            MAP[i] += maps[i]

        test_user_cnt += 1
        if test_user_cnt >= test_num:
            break
    
    print("recall sum : {}".format(sum(HIT)))
    print("total user : {}".format(test_user_cnt))
    HIT = np.array(HIT) / test_user_cnt
    MAP = np.array(MAP) / test_user_cnt

    return HIT, MAP

# unit : km
def haversine(la1, lo1, la2, lo2):
    lo1, la1, lo2, la2 = map(radians, [lo1, la1, lo2, la2])
    lo_d = lo2 - lo1
    la_d = la2 - la1
    a = sin(la_d / 2) ** 2 + cos(la1) * cos(la2) * sin(lo_d / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def poi_category_relation(dataset="NYC"):
    file_path = "./data/" + dataset + "/3_sort_total_u10p10_" + dataset + ".txt"
    poi2cate = {}
    with open(file_path, "r", encoding="utf-8") as rf:
        for line in rf:
            _, pid, _, _, _, _, cid, _ = line.strip().split("\t")
            pid, cid = int(pid), int(cid)
            if pid not in poi2cate:
                poi2cate[pid] = cid
    return poi2cate

def user_social_network(dataset="NYC", simi=0.2):
    file_path = "./data/" + dataset + "/3_sort_total_u10p10_" + dataset + ".txt"
    user_poi_dict = {}
    with open(file_path, "r", encoding="utf-8") as rf:
        for line in rf:
            uid, pid, lat, lon, timestamp, did, cid, utc = line.strip().split("\t")
            uid, pid = int(uid) - 1, int(pid) - 1
            if uid not in user_poi_dict:
                user_poi_dict[uid] = set()
            user_poi_dict[uid].add(pid)
    
    U = len(user_poi_dict)
    user_adj = np.zeros((U, U))

    for i in range(U):
        for j in range(i + 1, U):
            set_i, set_j = user_poi_dict[i], user_poi_dict[j]
            sim = float(len(set_i & set_j) / min(len(set_i), len(set_j)))
            if sim >= simi:
                user_adj[i, j] = user_adj[j, i] = 1
    for k in user_poi_dict.keys():
        user_poi_dict[k] = list(user_poi_dict[k])
    
    return user_adj, user_poi_dict

def geographical_info_build(poi_num, dataset="NYC", limit=3):
    print("build POI distance matrix...")
    t0 = time.time()

    file_path = "./data/" + dataset + "/3_sort_total_u10p10_" + dataset + ".txt"

    poi_lat_lon = defaultdict(tuple)
    with open(file_path, "r", encoding="utf-8") as rf:
        for line in rf:
            uid, pid, lat, lon, timestamp, did, cid, utc = line.strip().split("\t")
            uid, pid, timestamp, cid = int(uid) - 1, int(pid) - 1, int(timestamp), int(cid) - 1
            lat, lon = float(lat), float(lon)
            if pid not in poi_lat_lon:
                poi_lat_lon[pid] = (lat, lon)

    cnt = 0
    poi_distance_matrix = np.zeros((poi_num, poi_num))
    poi_distance_adj = np.zeros((poi_num, poi_num))
    poi_neighbors = {}
    src, dst = [], []

    poi_attention_coefficient = np.zeros((poi_num, poi_num))

    for i in range(poi_num):
        if i not in poi_neighbors:
            poi_neighbors[i] = []
        for j in range(i + 1, poi_num):
            (la1, lo1), (la2, lo2) = poi_lat_lon[i], poi_lat_lon[j]
            dist = haversine(la1, lo1, la2, lo2)
            poi_distance_matrix[i, j] = poi_distance_matrix[j, i] = dist
            
            if j not in poi_neighbors:
                poi_neighbors[j] = []
            
            if dist != 0.0 and dist <= limit:
                poi_neighbors[i].append(j)
                poi_neighbors[j].append(i)
                poi_distance_adj[i, j] = poi_distance_adj[j, i] = 1
                src.append(i)
                dst.append(j)
                cnt += 1

    for i in range(poi_num):
        neighbors_list = poi_neighbors[i]
        neighbors_dist = [poi_distance_matrix[i, x] for x in neighbors_list]
        exp_dist_sum = np.sum(np.exp(neighbors_dist))
        for nei_j in neighbors_list:
            nei_j_exp = np.exp(poi_distance_matrix[i, nei_j])
            
            poi_attention_coefficient[i, nei_j] = nei_j_exp / exp_dist_sum
            if i == 0 and nei_j == 5:
                print(poi_attention_coefficient[i, nei_j])

    src = torch.tensor(np.array(src))
    dst = torch.tensor(np.array(dst))
    poi_adj = dgl.graph((src, dst))
    poi_adj = dgl.to_simple(poi_adj)
    poi_adj = dgl.to_bidirected(poi_adj)
    poi_adj = dgl.add_self_loop(poi_adj)

    print("Finish building POI distance matrix : {}s".format(time.time() - t0))
    print("Sparsity Rate : {:.10f}".format(cnt / (poi_num * poi_num)))
    return poi_adj, poi_neighbors, poi_attention_coefficient
