# Importing All the important libraries and Functionalities

import os, sys, pdb, numpy as np, scipy.sparse as sp, random
import argparse, codecs, pickle, time, json, uuid
import networkx as nx
import logging, logging.config
import tensorflow as tf
import time

from collections import defaultdict as ddict
from pprint import pprint

np.set_printoptions(precision=4)

# Creating Functions for the initialization of the Weights, Matrix and Bias

def set_gpu(gpus):
        
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def debug_nn(res_list, feed_dict):
                
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
        res = sess.run(res_list, feed_dict = feed_dict)
        return res

def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
                index.append(int(line.strip()))
        return index


def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

# Preparing the operations to perform over datasets to make it ready for the calculations

def load_data(dataset_str, args):
        
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        if 'nell' in dataset_str:
                data_dict = pickle.load(open('./data/{}_data.pkl'.format(dataset_str), 'rb'), encoding='latin1')
                x, y, tx, ty, allx, ally, graph = data_dict['x'], data_dict['y'], data_dict['tx'], data_dict['ty'], data_dict['allx'], data_dict['ally'], data_dict['graph']

                index = list(range(allx.shape[0])) + data_dict['test.index']
                remap = {x: x for x in range(allx.shape[0])}
                remap.update({i+allx.shape[0]: x for i, x in enumerate(data_dict['test.index'])})
                remap_inv = {v: k for k, v in remap.items()}

                graph_new = ddict(list)
                for key, val in graph.items():
                        if key not in remap_inv: continue
                        graph_new[remap_inv[key]] = [remap_inv[v] for v in val if v in remap_inv]

                graph = graph_new
                test_idx_reorder = [remap_inv[x] for x in data_dict['test.index']]
        else:
                for i in range(len(names)):
                        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                                if sys.version_info > (3, 0):
                                        objects.append(pickle.load(f, encoding='latin1'))
                                else:
                                        objects.append(pickle.load(f))

                x, y, tx, ty, allx, ally, graph = tuple(objects)
                test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))

        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
                # Fix citeseer dataset (there are some isolated nodes in the graph)
                # Find isolated nodes, add them as zero-vecs into the right position
                test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
                tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
                tx_extended[test_idx_range-min(test_idx_range), :] = tx
                tx = tx_extended
                ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
                ty_extended[test_idx_range-min(test_idx_range), :] = ty
                ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask   = sample_mask(idx_val, labels.shape[0])
        test_mask  = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val   = np.zeros(labels.shape)
        y_test  = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

# Creating the Sparce Matrix and Adjacency Matrix from the features of Data set and normalizing  
    
def sparse_to_tuple(sparse_mx):
        """Convert sparse matrix to tuple representation."""
        def to_tuple(mx):
                if not sp.isspmatrix_coo(mx):
                        mx = mx.tocoo()
                coords = np.vstack((mx.row, mx.col)).transpose()
                values = mx.data
                shape = mx.shape
                return coords, values, shape

        if isinstance(sparse_mx, list):
                for i in range(len(sparse_mx)):
                        sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
                sparse_mx = to_tuple(sparse_mx)

        return sparse_mx


def preprocess_features(features, noTuple=False):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)

        if noTuple:     return features
        else:           return sparse_to_tuple(features)

def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# Creating the Adjacency and Identity Matrix and here we have added the clustering Co-efficient matrix
    
def preprocess_adj(adj, noTuple=False):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        
        aa = sp.lil_matrix(adj).toarray()
        g = nx.from_numpy_matrix(aa, parallel_edges=True)
        cluster = nx.clustering(g)
        diag = np.diag(g)
#         adj_normalized = normalize_adj(adj + diag) # For "Cora" Datasest only
        
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))  # For "CoraML", "Pubmed" and "Citeseer" Datasests
        
        if noTuple:     return adj_normalized
        else:           return sparse_to_tuple(adj_normalized)


def get_ind_from_adj(adj):
        lens = [len(list(np.nonzero(row)[0])) for row in adj]
        ind  = np.zeros((adj.shape[0], np.max(lens)), dtype=np.int64)
        mask = np.zeros((adj.shape[0], np.max(lens)), dtype=np.float32)

        for i, row in enumerate(adj):
                J = np.nonzero(row)[1]
                for pos, j in enumerate(J):
                        ind[i][pos]  = j
                        mask[i][pos] = 1

        return ind, mask

# Main Graph Convolutional Layer

class ConfGCN(object):

    def load_data(self):

#         print("loading data")
        self.adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask = load_data(self.p.data, self.p)
        self.features = preprocess_features(self.features, noTuple=False)
        self.adj = preprocess_adj(self.adj, noTuple=True).todense()
        self.adj_ind, self.adj_ind_mask = get_ind_from_adj(self.adj)

        self.num_nodes    = self.features[2][0]
        self.input_dim    = self.features[2][1]
        self.output_dim   = self.y_train.shape[1]

        # Label mask
        self.label_cond = np.zeros((self.num_nodes), np.bool)
        for i in range(self.num_nodes):
            if np.sum(self.y_train[i]) != 0:
                self.label_cond[i] = 1

        self.placeholders = {
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(self.features[2], dtype=tf.int64)), 
            # features[2] = shape of the input
            'labels': tf.placeholder(tf.float32,   shape=(None, self.y_train.shape[1])),   # batch x 7(num_classes)
            'labels_mask': tf.placeholder(tf.int32),
            'adj_ind':  tf.placeholder(tf.int32),
            'adj_ind_mask': tf.placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0.,   shape=()),# Dropout
            'num_features_nonzero': tf.placeholder(tf.int32) # helper variable for sparse dropout
        }

    def create_feed_dict(self, split='train'):
        feed = {}

        feed[self.placeholders['features']]= self.features
        feed[self.placeholders['adj_ind']] = self.adj_ind
        feed[self.placeholders['adj_ind_mask']] = self.adj_ind_mask
        feed[self.placeholders['num_features_nonzero']] = self.features[1].shape

        if split == 'train':
            feed[self.placeholders['labels']] = self.y_train
            feed[self.placeholders['labels_mask']] = self.train_mask
            feed[self.placeholders['dropout']] = self.p.drop
        elif split == 'test':
            feed[self.placeholders['labels']] = self.y_test
            feed[self.placeholders['labels_mask']] = self.test_mask
            feed[self.placeholders['dropout']] = 0.0
        else:
            feed[self.placeholders['labels']] = self.y_val
            feed[self.placeholders['labels_mask']] = self.val_mask
            feed[self.placeholders['dropout']] = 0.0

        return feed

    def sparse_dropout(self, x, keep_prob, noise_shape):
        
        random_tensor  = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask   = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out        = tf.sparse_retain(x, dropout_mask)
        return pre_out * (1./keep_prob)

    def matmul(self, a, b, is_sparse=False):
       
        if is_sparse: return tf.sparse_tensor_dense_matmul(a, b)
        else: return tf.matmul(a, b)

    def dropout(self, inp, dropout, num_feat_nonzero=0, is_sparse=False):
       
        if is_sparse: return self.sparse_dropout(inp, 1 - dropout, num_feat_nonzero)
        else: return tf.nn.dropout(inp, 1-dropout)

# Creating Layers Basic Functionality
        
    def GCNLayer(self, gcn_in, adj_ind, adj_ind_mask, 
                 input_dim, output_dim, act, dropout, num_features_nonzero, input_sparse=False, name='GCN'):
       

        with tf.variable_scope('{}_vars'.format(name)) as scope:
            wts  = tf.get_variable('weights', [input_dim, output_dim], 
                                   initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)
            bias = tf.get_variable('bias',    [output_dim], 
                                   initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)

        gcn_in   = self.dropout(gcn_in, dropout, num_features_nonzero, is_sparse=input_sparse)
        node_act = self.matmul(gcn_in,  wts, is_sparse=input_sparse)

        f_vecs   = tf.nn.embedding_lookup(self.mu, adj_ind)
        f_diff   = f_vecs - tf.expand_dims(self.mu, axis=1)
        f_diff   = f_diff * tf.expand_dims(adj_ind_mask, axis=2)

        sig_vecs = tf.nn.embedding_lookup(self.sig, adj_ind)
        sig_sum  = sig_vecs + tf.expand_dims(self.sig, axis=1)
        sig_sum  = sig_sum  * tf.expand_dims(adj_ind_mask, axis=2)

        dist     = tf.reduce_sum(f_diff * (f_diff * sig_sum), axis=2) + self.p.bias
        dist     = 1 / dist
        dist     = tf.exp(dist - tf.reduce_max(dist, axis=1, keepdims=True)) * adj_ind_mask
        dist     = dist / tf.reduce_sum(dist, axis=1, keepdims=True)

        act_vecs = tf.nn.embedding_lookup(node_act, adj_ind)
        act_vecs = act_vecs * tf.expand_dims(adj_ind_mask, axis=2)

        final_act = tf.reduce_sum(act_vecs * tf.expand_dims(dist, axis=2), axis=1)
        gcn_out   = final_act

        return gcn_out
    
# A New Suggested layer having the Dense Structure to change the Graph Convolutional Weighing Scheme
    
    def Weighted_Sum_Layer(self, gcn_in, adj_ind, adj_ind_mask, 
                 input_dim, output_dim, act, dropout, num_features_nonzero, input_sparse=False, name='GCN'):
       

        with tf.variable_scope('{}_vars'.format(name)) as scope:
            wts  = tf.get_variable('weights', [input_dim, output_dim], 
                                   initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)
            bias = tf.get_variable('bias',    [output_dim], 
                                   initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)

        node_act = self.matmul(gcn_in,  wts) + self.p.bias
        

        return node_act
    
    
# ConfGCN Model having normally 3 Layered Structure, 
# but here we have added the Another Hidden Layer having Dense Structure


    def add_model(self):
        
        self.layers, self.activations = [], []

        with tf.variable_scope('main_variables') as scope:
            self.mu  = tf.get_variable('mu',  [self.num_nodes,  self.output_dim], 
                                       initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)
            # Label distribution for each node
            self.sig = tf.get_variable('sig', [self.num_nodes,  self.output_dim], 
                                       initializer=tf.constant_initializer(1.0), regularizer=self.regularizer)
            # Inverse of co-variance matrix

        self.mu  = tf.nn.softmax(self.mu, axis = 1)# Makes mu into a distribution
        self.sig = tf.nn.elu(self.sig)# Imposes soft non-negative constraint on co-variance matrix

        gcn1_out = self.GCNLayer(
                gcn_in                  = self.placeholders['features'],
                adj_ind                 = self.placeholders['adj_ind'],
                adj_ind_mask            = self.placeholders['adj_ind_mask'],
                input_dim               = self.input_dim,
#                 output_dim              = self.p.gcn_dim,
                output_dim              = 16,        # 16 for "Pubmed", "Citeseer", and "CoraML" and 32 for "Cora" and "Citeseer"
                act                     = tf.nn.relu6,    #Relu6 or Selu provides efficient results
                dropout                 = self.placeholders['dropout'],
                num_features_nonzero    = self.placeholders['num_features_nonzero'],
                input_sparse            = True,
                name                    = 'GCN_1'
            )
        
# If New Convolutional Layer or another Input layer needs to be added, below code needs to be added:
        
#         gcn3_out = self.GCNLayer(
# #                 gcn_in                  = gcn1_out,     # For another Convolutional Layers, i.e. 4 Layered Structure
#                 gcn_in                  = self.placeholders['features'],   # For another Input Layer, summation opeartion
#                 adj_ind                 = self.placeholders['adj_ind'],
#                 adj_ind_mask            = self.placeholders['adj_ind_mask'],
#                 input_dim               = self.input_dim,     # For another Input Layer, summation opeartion
# #                 input_dim               = 16,   # For another Convolutional Layers, i.e. 4 Layered Structure
#                 output_dim              = 16,     # 16 nodes for input layer and 32 nodes for another conv. layer
#                 act                     = tf.nn.selu,
#                 dropout                 = self.placeholders['dropout'],
#                 num_features_nonzero    = self.placeholders['num_features_nonzero'],
#                 input_sparse            = True,
#                 name                    = 'GCN_3'
#             )

        gcn2_out = self.GCNLayer(
                gcn_in                  = gcn1_out,   # Simple 3 layered Structure
#                 gcn_in                  = gcn3_out, # Additional Conv. layer, 4 Layer Structure
#                 gcn_in                  = gcn1_out+gcn3_out,  # When 2 Input Layers are considered

            # When 2 Input Layers are considered for Canonical form
            
#                 gcn_in                  = tf.scalar_mul(0.8, gcn1_out) + tf.scalar_mul(0.2,gcn3_out), 
                adj_ind                 = self.placeholders['adj_ind'],
                adj_ind_mask            = self.placeholders['adj_ind_mask'],
                input_dim               = 16,  # 16 nodes for input layer and 32 nodes for another conv. layer
                output_dim              = self.output_dim,
                act                     = lambda x: x,
                dropout                 = self.placeholders['dropout'],
                num_features_nonzero    = self.placeholders['num_features_nonzero'],
                input_sparse            = False,
                name                    = 'GCN_2'
            )

        nn_out = gcn2_out
        return nn_out

    def get_accuracy(self, nn_out):
        

        correct_prediction  = tf.equal(tf.argmax(nn_out, 1), tf.argmax(self.placeholders['labels'], 1))
        # Identity position where prediction matches labels
        accuracy_all  = tf.cast(correct_prediction, tf.float32)# Cast result to float
        mask  = tf.cast(self.placeholders['labels_mask'], dtype=tf.float32)# Cast mask to float
        mask /= tf.reduce_mean(mask)# Compute mean of mask
        accuracy_all *= mask # Apply mask on computed accuracy

        return tf.reduce_mean(accuracy_all)


    def loss_smooth(self, adj_ind, adj_ind_mask):
       
        mu_vecs  = tf.nn.embedding_lookup(self.mu, adj_ind)
        mu_diff  = (mu_vecs - tf.expand_dims(self.mu, axis=1)) * tf.expand_dims(adj_ind_mask, axis=2)

        sig_vecs = tf.nn.embedding_lookup(self.sig, adj_ind)
        sig_sum  = (sig_vecs + tf.expand_dims(self.sig, axis=1)) * tf.expand_dims(adj_ind_mask, axis=2)

        loss     = tf.reduce_sum(mu_diff * (mu_diff * sig_sum))

        return loss

    def loss_label(self):
        
        node_ind = tf.squeeze(tf.where(tf.not_equal(self.placeholders['labels_mask'], 0)), axis=1)

        mu_vecs  = tf.gather(self.mu, node_ind)
        y_actual = tf.gather(self.placeholders['labels'], node_ind)
        mu_diff  = y_actual - mu_vecs

        sig_vecs = tf.gather(self.sig, node_ind) + self.p.gamma
        loss     = tf.reduce_sum(mu_diff * ((mu_diff * sig_vecs)))

        return loss

    def loss_const(self, nn_out):
        
        pred  = tf.nn.softmax(nn_out)
        loss  = tf.square(pred - self.mu)
        loss  = loss * tf.expand_dims(tf.cast(self.placeholders['labels_mask'], tf.float32), axis=1)
        return tf.reduce_sum(loss)

    def loss_reg(self):
        
        return tf.reduce_sum(tf.where(self.sig < 0, -self.sig, tf.zeros_like(self.sig)))

# Making the regularization and Optimization in Loss Function with Version 2
    
    def add_loss_op(self, nn_out):
       
        loss  = 0
        temp       = tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_out, labels=self.placeholders['labels'])
        
        # Compute cross entropy loss
        mask       = tf.cast(self.placeholders['labels_mask'], dtype=tf.float32)# Cast masking from boolean to float

        loss += self.p.l_cross * tf.reduce_sum(temp * mask) / tf.reduce_sum(mask)
        loss += 1/4 * self.p.l_smooth * self.loss_smooth(self.placeholders['adj_ind'], self.placeholders['adj_ind_mask'])
        loss += 1/2 * self.p.l_label * self.loss_label()
        loss += self.p.l_const * self.loss_const(nn_out)
        loss += self.p.l_reg * self.loss_reg()

        if self.regularizer != None:
            loss += tf.contrib.layers.apply_regularization(self.regularizer, 
                                                           tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        return loss

    def add_optimizer(self, loss, isAdam=True):
        
        with tf.name_scope('Optimizer'):
            if isAdam:  optimizer = tf.train.AdamOptimizer(self.p.lr)
            else:       optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
            train_op  = optimizer.minimize(loss)

        return train_op

# Loading the Data-set and Training, Validation and Testing    

    def __init__(self, params):
       
        self.p  = params

        if self.p.l2 == 0.0:    self.regularizer = None
        else:  self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

        self.load_data()

        nn_out  = self.add_model()
        self.loss = self.add_loss_op(nn_out)
        self.accuracy = self.get_accuracy(nn_out)

        self.train_op = self.add_optimizer(self.loss)

        self.merged_summ = tf.summary.merge_all()
        self.summ_writer = None


    def evaluate(self, split='valid'):
        time_test = time.time()
        feed_dict = self.create_feed_dict(split=split)  # Defines the feed_dict to be fed to NN
        loss, acc = [model.loss, model.accuracy] # Computer loss and accuracy
        out_test = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        print("Testing Accuracy", "{:.2f}".format((out_test[1])*100), 
              "% and Execution time : ", "{:.5f}".format(time.time()-time_test))
        return loss, acc # return loss, accuracy

    def run_epoch(self, sess, epoch, shuffle=True):
        t_test = time.time()
        feed_dict = self.create_feed_dict(split='train')

        outs = sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)# Training step
#         cost, acc = self.evaluate(split='valid')# Computer Validation performance

        print("Training Accuracy : ","{:.2f}".format(outs[2]*100),
              "% and Loss : ", "{:.5f}".format(outs[1]/100),
              "Execution time : ", "{:.5f}".format(time.time()-t_test))

    def fit(self, sess):
        epoch_list = []
        for epoch in range(250):
            train_loss = self.run_epoch(sess, epoch)
            epoch_list.append(epoch+1)
            print(epoch+1)

        print("Test set results:")
        test_cost, test_acc  = self.evaluate(split='test')        

tf.reset_default_graph()

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='Confidence-based GCN')

    parser.add_argument('-data',    default='cora_ml',help='Dataset to use') # cora, citeseer, cora_ml, pubmed
    parser.add_argument('-gpu',     default='0',help='GPU to use')
    parser.add_argument('-name',    default='test',help='Name of the run')

    parser.add_argument('-lr',      default=0.01,type=float,help='Learning rate')
    parser.add_argument('-epochs',  default=500,type=int,help='Max epochs')
    parser.add_argument('-l2',      default=0.01,type=float,help='L2 regularization')
    parser.add_argument('-opt',     default='adam',             help='Optimizer to use for training')
    parser.add_argument('-gcn_dim', default=16,type=int,       help='GCN hidden dimension')
    parser.add_argument('-drop',    default=0.5,type=float,     help='Dropout for full connected layer')

    parser.add_argument('-l_cross', default=1, type=float,help='L_cross value')
    parser.add_argument('-l_smooth',default=1, type=float,help='L_smooth value')
    parser.add_argument('-l_label', default=0, type=float,help='L_label value')
    parser.add_argument('-l_const', default=10, type=float,help='L_const value')
    parser.add_argument('-l_reg', default=1, type=float,help='L_reg value')
    parser.add_argument('-gamma', default=3, type=float,help='Gamma value')
    parser.add_argument('-bias', default=0.1, type=float,help='bias value')

    parser.add_argument('-restore',  action='store_true',        help='Restore from the previous best saved model')
    parser.add_argument('-eval',   action='store_true',        help='Set evaluation only mode')
    parser.add_argument('-manual_param', action='store_true',        help='Set evaluation only mode')

    parser.add_argument('-logdir',  dest="log_dir",        default='./log/',      help='Log directory')
    parser.add_argument('-config',  dest="config_dir",     default='./config/',    help='Config directory')

    args = parser.parse_known_args()[0]

    # Not changing name when restoring previously saved model
    if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S") + '_' + str(uuid.uuid4())[:8]

    if not args.manual_param:
        params = json.load(open('hyperparams.json'))
        for key, val in params[args.data].items():
            exec('args.{}={}'.format(key, val))

    # Evaluation only model (no training)
    if args.eval: args.epochs = 0

    # Set GPU
    set_gpu(args.gpu)

    # Create model
    model = ConfGCN(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess)# Start training
