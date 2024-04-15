# This code is licensed under the GNU General Public License v3.0
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# *********************************************
#
# Implemenation of "Natural Learning"
#
# website: www.natural-learning.cc
# Author: Hadi Fanaee-T
# Associate Professor of Machine Learning
# School of Information Technology
# Halmstad University, Sweden
# Email: hadi.fanaee@hh.se
#
#
# Please cite the following paper if you use the code
#
# *********************************************
# Hadi Fanaee-T, "Natural Learning", arXiv:2404.05903
# https://arxiv.org/abs/2404.05903
#
# *********************************************
# BibTeX
# *********************************************
#
# @article{fanaee2024natural,
#   title={Natural Learning},
#   author={Fanaee-T, Hadi},
#   journal={arXiv preprint arXiv:2404.05903},
#   year={2024}
#}
#

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class LSH:
    def __init__(self, X=None):
        if X is not None:
            n, m = X.shape
            self.num_hash_functions = 10
            self.hash_size = 100
            np.random.seed(42)
            self.hash_functions = np.random.randn(m, self.num_hash_functions)
            hash_codes = np.sign(np.dot(X, self.hash_functions))
            self.hash_tables = [None] * self.hash_size
            for i in range(n):
                hash_code = hash_codes[i, :]
                hash_index = np.mod(np.sum(hash_code), self.hash_size).astype(int)
                if self.hash_tables[hash_index] is None:
                    self.hash_tables[hash_index] = [i]
                else:
                    self.hash_tables[hash_index].append(i)
        else:
            self.hash_functions = None
            self.hash_tables = None
            
    def query(self, query_point, X):
        query_hash_code = np.sign(np.dot(query_point, self.hash_functions))
        query_hash_index = np.mod(np.sum(query_hash_code), self.hash_size).astype(int)
        candidate_neighbors = self.hash_tables[query_hash_index]
        if not candidate_neighbors: 
            return -1
        else:
            best_distance = np.inf
            best_neighbor = -1
            for i in candidate_neighbors:
                candidate_point = X[i, :]
                distance = np.linalg.norm(candidate_point - query_point)
                if distance < best_distance and distance != 0:
                    best_distance = distance
                    best_neighbor = i
        return best_neighbor
    

class  NL:

    def fit(self, X,y):
        n, p = X.shape
        y=y.flatten()
        M = np.arange(p)
        ids0 = np.where(y == 0)[0]
        ids1 = np.where(y == 1)[0]
        L = 0
        e_best = np.inf
        while True:
            L += 1
            Mdl0 = LSH(X[ids0][:, M])
            Mdl1 = LSH(X[ids1][:, M])

            for i in range(n):
                s = Mdl0.query(X[i, M],X[ids0][:, M])
                if s != -1:
                    s = ids0[s]
                    o = Mdl1.query(X[i, M],X[ids1][:, M])
                    if o != -1:
                        o = ids1[o]
                        if y[i] == 1:
                            tmp_o = o
                            o = s
                            s = tmp_o
                        vs = np.abs(X[s, M] - X[i, M])
                        vo = np.abs(X[o, M] - X[i, M])
                        v = vo - vs
                        C = M[np.where(v > 0)[0]]
                        if C.size > 1:
                            yt = y[[s, o]]
                            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X[[s, o]][:, C])
                            distances, indices = nbrs.kneighbors(X[:, C])
                            yhat = yt[indices].flatten()
                            e = np.sum(yhat != y)
                            if e < e_best:
                                C_best = C
                                s_best = s
                                o_best = o
                                e_best = e
                                L_best = L
                                last_e_best = e_best
                                print(f"Layer={L} Sample={i}/{n} Prototype=[{s_best},{o_best}], NumFeatures={len(C_best)}/{len(M)}, Error={e_best/n:.4f}")

            if len(C_best) == len(M):
                break
            else:
                M = C_best
                e_best = np.inf
        
        best_prototype = f"[{s_best} (class {y[s_best]}), {o_best} (class {y[o_best]})]"
        features = ' '.join(map(str, M))
        print(f"Best Prototypes={best_prototype}, BestError={last_e_best/n:.4f}, Prototype Features=[{features}]")

        self.Mdl = {}
        self.Mdl['PrototypeSampleIDs'] = [s_best, o_best]
        self.Mdl['PrototypeFeatureIDs'] = M
        self.Mdl['Error'] = last_e_best / n
        self.Mdl['NLayers'] = L_best
        self.Mdl['MX'] = X[[s_best, o_best]][:, M]
        self.Mdl['My'] = y[[s_best, o_best]]

    def predict(self, X_test):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.Mdl["MX"])
        distances, indices = nbrs.kneighbors(X_test[:, self.Mdl["PrototypeFeatureIDs"]])
        yt=self.Mdl["My"]
        y_pred = yt[indices].flatten()
        return y_pred


X_train = pd.read_csv('X_train.csv', header=None)
X_test = pd.read_csv('X_test.csv', header=None)
y_train = pd.read_csv('y_train.csv', header=None)
y_test = pd.read_csv('y_test.csv', header=None)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
clf=NL()
clf.fit(X_train,y_train)
y_pred_test=clf.predict(X_test)
err_test = np.sum(y_pred_test != y_test.flatten())/y_test.shape[0]
print('Test Error = '+str(err_test))
