
import math
import torch

def kmeans(X : 'BCT', K = 5, num_iter = 10, init = 'random', generator = None) -> '(BT, BCK)':
    if torch.is_tensor(init):
        C = init
    elif init == 'random':
        C = X.gather(-1, torch.rand(X.shape[0], X.shape[-1], device = X.device, generator = generator).argsort(-1).narrow(-1, 0, K).unsqueeze(1).expand(-1, X.shape[1], -1))
    elif init == 'k-means++':
        Z, C = kmeans_plusplus(X, K = K, generator = generator)

    for i in range(num_iter):
        D = cdist_squared(A, C);
        Z = D.argmin(dim = -1)
        bincount = Z.new_zeros(len(Z), K).scatter_add_(-1, Z, torch.ones_like(Z)).unsqueeze(1)
        C.zero_().scatter_add_(-1, Z.unsqueeze(1).expand(-1, X.shape[1], -1), A).div_(bincount.clamp(min = 1))
    
    return Z, C

def kmeans_ll(X : 'BCT', K, num_iter = None, generator = None) -> '(BT, BCK)':
    # https://github.com/zxytim/kmeansII/blob/master/src/kmeansII.cc#L137
    # https://nbviewer.jupyter.org/github/jtappler/ScalableKMeansPlusPlus/blob/master/ScalableKMeansPlusPlus.ipynb
    pass

def kmc2(X : 'BCT', K, num_iter = None, generator = None) -> '(BT, BCK)':
    # https://github.com/obachem/kmc2/blob/master/kmc2.pyx#L53 
    pass

def kmeans_plusplus(X : 'BCT', K, num_iter = None, generator = None) -> '(BT, BCK)':
    # https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/cluster/_kmeans.py#L604
    if num_iter is None:
        num_iter = 2 + int(math.log(K))
    
    Z = torch.full((len(X), K), -1, dtype=int)
    C = torch.empty((*X.shape[:2], K), dtype=X.dtype)
    
    Z[:, 0] = torch.randint(X.shape[-1], (len(X), ), generator = generator)
    C[:, :, 0] = X.gather(-1, Z[:, 0][:, None, None].expand(-1, X.shape[1], -1)).squeeze(-1)

    closest_dist_sq = cdist_squared(C[..., :1], X).squeeze(1) 
    for i in range(1, K):
        candidate_ids = torch.multinomial(closest_dist_sq, num_iter, generator = generator)
        distance_to_candidates = cdist_squared(X.gather(-1, candidate_ids.unsqueeze(1).expand(-1, X.shape[1], -1)), X)
        torch.minimum(distance_to_candidates, closest_dist_sq.unsqueeze(1), out=distance_to_candidates)

        best_candidate = distance_to_candidates.sum(dim = -1).argmin(dim = 1)
        closest_dist_sq = distance_to_candidates.gather(1, best_candidate[:, None, None].expand(-1, -1, distance_to_candidates.shape[-1])).squeeze(1)
        Z[:, i] = candidate_ids.gather(1, best_candidate[:, None]).squeeze(1)
        C[:, :, i] = X.gather(-1, Z[:, i][:, None, None].expand(-1, X.shape[1], -1)).squeeze(-1)

    return Z, C

def cdist_squared(A : 'BCT', B : 'BCK', eps = 1e-4) -> 'BTK':
    normAsq = A.pow(2).sum(dim = 1, keepdim = True)
    normBsq = B.pow(2).sum(dim = 1, keepdim = True)
    res = torch.baddbmm(normBsq, A.transpose(-2, -1), B, alpha = -2).add_(normAsq.transpose(-2, -1))
    return res.clamp_(min = 0)

if __name__ == '__main__':
    X = torch.rand(16, 128, 13)
    K = 12

    C, Z = _kmeans_plusplus(X, K)
