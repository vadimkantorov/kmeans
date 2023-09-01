import math
import torch
import torch.nn.functional as F

def kmeans2d(X : 'BCHW', K = 12, num_iter = 10, init = 'random', generator = None, posencdim = 2, radius = 0.1, mask = 'bilateral') -> '(BHW, BCK)':
    assert X.is_floating_point()

    X_shape = X.shape
    posenc = torch.stack(torch.meshgrid(torch.arange(X.shape[-2], dtype = torch.float32), torch.arange(X.shape[-1], dtype = torch.float32), indexing = 'ij'), dim = -3)
    X = torch.cat([posenc.unsqueeze(0).expand(X.shape[0], -1, -1, -1), X], dim = -3).flatten(start_dim = -2)
        
    assert posenc.shape[-3] == posencdim
    assert init == 'random'
    #I, C = torch.randint(0, K, (X.shape[0], X.shape[-1]), device = X.device, dtype = torch.int64, generator = generator), X.gather(-1, torch.rand(X.shape[0], X.shape[-1], device = X.device, generator = generator).argsort(-1).narrow(-1, 0, K).unsqueeze(1).expand(-1, X.shape[1], -1))
    
    I, C = torch.randperm(X.shape[-1], device = X.device).unsqueeze(0), X.gather(-1, torch.rand(X.shape[0], X.shape[-1], device = X.device, generator = generator).argsort(-1).narrow(-1, 0, K).unsqueeze(1).expand(-1, X.shape[1], -1))
    
    M = torch.zeros(X.shape[0], X.shape[-1], C.shape[-1], device = X.device, dtype = torch.bool)
    J = F.unfold(torch.arange(X.shape[-1], device = X.device).view(X_shape[-2:]).view(torch.float64)[None, None, :, :], kernel_size = 2 * radius + 1, padding = radius).view(torch.int64).expand(X.shape[0], -1, -1).transpose(-1, -2) if mask == 'rep' else None
    
    history = []
    for i in range(num_iter):
        if mask == 'bil':
            M = torch.lt(cdist_squared(X[:, :posencdim], C[:, :posencdim]), radius * radius, out = M)
            
            D = torch.where(M, cdist_squared(X[:, posencdim:], C[:, posencdim:]), float('inf'))
            I = D.argmin(dim = -1)
        
        elif mask == 'rep':
            JJ = I.unsqueeze(-1).expand(-1, -1, (2 * radius + 1) ** 2).gather(-2, J)
            #CC = C.transpose(-1, -2).gather(-2, JJ.flatten(start_dim = -2).unsqueeze(-1).expand(-1, -1, C.shape[-2])).unflatten(-2, JJ.shape[-2:])
            #I = JJ.gather(-1, (CC * X.transpose(-1, -2).unsqueeze(-2)).sum(dim = -1).argmax(dim = -1, keepdim = True)).squeeze(-1)
            M.zero_().scatter_(-1, JJ, True)
            D = torch.where(M, cdist_squared(X[:, posencdim:], C[:, posencdim:]), float('inf'))
            I = D.argmin(dim = -1)
        
        bincount = torch.zeros((len(I), K), dtype = I.dtype, device = I.device).scatter_add_(-1, I, torch.ones_like(I)).unsqueeze(1)
        C.zero_().scatter_add_(-1, I.unsqueeze(1).expand(-1, X.shape[1], -1), X).div_(bincount.clamp_(min = 1))
        history.append((I.unflatten(-1, X_shape[-2:]).clone(), C[:, posencdim:].clone(), C[:, :posencdim].clone()))
    
    return I.unflatten(-1, X_shape[-2:]), C[:, posencdim:], C[:, :posencdim], history

        

def kmeans1d(X : 'BCT', K = 5, num_iter = 10, init = 'random', generator = None) -> '(BT, BCK)':
    if torch.is_tensor(init):
        I, C = None, init
    elif init is None or init == 'random':
        I, C = None, X.gather(-1, torch.rand(X.shape[0], X.shape[-1], device = X.device, generator = generator).argsort(-1).narrow(-1, 0, K).unsqueeze(1).expand(-1, X.shape[1], -1))
    elif init == 'k-means++':
        I, C = kmeans_plusplus(X, K = K, generator = generator)

    for i in range(num_iter):
        D = cdist_squared(X, C);
        I = D.argmin(dim = -1)
        bincount = torch.zeros((len(I), K), dtype = I.dtype, device = I.device).scatter_add_(-1, I, torch.ones_like(I)).unsqueeze(1)
        C.zero_().scatter_add_(-1, I.unsqueeze(1).expand(-1, X.shape[1], -1), X).div_(bincount.clamp_(min = 1))
    
    return I, C

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
    
    I = torch.full((len(X), K), -1, dtype=int)
    C = torch.empty((*X.shape[:2], K), dtype=X.dtype)
    
    I[:, 0] = torch.randint(X.shape[-1], (len(X), ), generator = generator)
    C[:, :, 0] = X.gather(-1, I[:, 0][:, None, None].expand(-1, X.shape[1], -1)).squeeze(-1)

    closest_dist_sq = cdist_squared(C[..., :1], X).squeeze(1) 
    for i in range(1, K):
        candidate_ids = torch.multinomial(closest_dist_sq, num_iter, generator = generator)
        distance_to_candidates = cdist_squared(X.gather(-1, candidate_ids.unsqueeze(1).expand(-1, X.shape[1], -1)), X)
        torch.minimum(distance_to_candidates, closest_dist_sq.unsqueeze(1), out=distance_to_candidates)

        best_candidate = distance_to_candidates.sum(dim = -1).argmin(dim = 1)
        closest_dist_sq = distance_to_candidates.gather(1, best_candidate[:, None, None].expand(-1, -1, distance_to_candidates.shape[-1])).squeeze(1)
        I[:, i] = candidate_ids.gather(1, best_candidate[:, None]).squeeze(1)
        C[:, :, i] = X.gather(-1, I[:, i][:, None, None].expand(-1, X.shape[1], -1)).squeeze(-1)

    return I, C

def cdist_squared(A : 'BCT', B : 'BCK', out = None) -> 'BTK':
    normAsq = A.pow(2).sum(dim = 1, keepdim = True)
    normBsq = B.pow(2).sum(dim = 1, keepdim = True)
    return torch.baddbmm(normBsq, A.transpose(-2, -1), B, alpha = -2, out = out).add_(normAsq.transpose(-2, -1)).clamp_(min = 0)

def pdist_squared(A : 'BCT', out = None) -> 'BTT':
    normAsq = A.pow(2).sum(dim = 1, keepdim = True)
    return torch.baddbmm(normAsq, A.transpose(-2, -1), A, alpha = -2, out = out).add_(normAsq.transpose(-2, -1)).clamp_(min = 0)

def remap_unique(X):
    return torch.stack([t.unique(return_inverse = True)[-1] for t in X.flatten(start_dim = 1)]).unflatten(-1, X.shape[1:])

