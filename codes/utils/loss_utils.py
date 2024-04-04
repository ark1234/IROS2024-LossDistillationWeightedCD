
'''
==============================================================

    0-------------------------------0
    |       Loss Functions          |
    0-------------------------------0

==============================================================

    Compute chamfer distance loss L1/L2

==============================================================
'''

import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.utils import fps_subsample

from geoopt.manifolds import PoincareBall as PoincareBallParent
from geoopt.manifolds.stereographic.math import _lambda_x, arsinh, tanh

MIN_NORM = 1e-15


chamfer_dist = chamfer_3DDist()


from hyptorch import nn as hypnn
from hyptorch.pmath import dist_matrix

def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    # import pdb; pdb.set_trace()
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def get_loss(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, gt)

    partial_matching = PM(partial, P3)

    loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e3
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses, [gt_2, gt_1, gt_c]







#loss_weighted_CDs
def get_loss1(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    # cdc = CD(Pc, gt_c)
    cdc = calc_cd_like_weighted(Pc, gt_c)
    # cd1 = CD(P1, gt_1)
    cd1 = calc_cd_like_weighted(P1, gt_1)
    # cd2 = CD(P2, gt_2)
    cd2 = calc_cd_like_weighted(P2, gt_2)
    # cd3 = CD(P3, gt)
    cd3 = calc_cd_like_weighted(P3, gt)

    # partial_matching = PM(partial, P3)
    partial_matching = calc_cd_one_side_like_weighted(partial, P3)

    loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e3
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses, [gt_2, gt_1, gt_c]



def calc_cd_like_weighted(p1, p2):

    # cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    dist1 = torch.clamp(dist1, min=1e-9)
    dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)











# ######wighted_extreme_value_sigma1.4_miu_0
    

    # scaler = torch.exp(-(2/3)*distances) * torch.exp(-torch.exp(-(2/3)*distances))
    scaler1 = 0.7143 * torch.exp(-(d1/1.4)) * torch.exp(-torch.exp(-(d1/1.4)))

    distances1 = scaler1 * d1


    scaler2 = 0.7143 * torch.exp(-(d2/1.4)) * torch.exp(-torch.exp(-(d2/1.4)))

    distances2 = scaler2 * d2

    # distances = torch.mean(distances)
    # distances = torch.sum(distances)
    # return distances
# ######wighted_extreme_value_sigma1.4_miu_0
    


# ######wighted_Chi-squared distribution k=3
 

    scaler1 = 0.3989422468 * (torch.sqrt(d1)) * torch.exp(- (d1/2))

    distances1 = scaler1 * d1

    scaler2 = 0.3989422468 * (torch.sqrt(d2)) * torch.exp(- (d2/2))

    distances2 = scaler2 * d2

    # distances = torch.mean(distances)
    # distances = torch.sum(distances)
    # return distances
# ######wighted_Chi-squared distribution k=3






######1023_Log-logistic alpha=5 beta=2
 
    # Numerator = 0.08 * distances
    # Denominator = (1+ (distances**2) * 0.04)**2
    scaler1 = (0.08 * d1) / ((1+ (d1**2) * 0.04)**2)

    distances1 = scaler1 * d1
    scaler2 = (0.08 * d2) / ((1+ (d2**2) * 0.04)**2)

    distances2 = scaler2 * d2


    # distances = torch.mean(distances)
    # distances = torch.sum(distances)
    # return distances
######1023_Log-logistic  alpha=5 beta=2




# ######wighted_gamma distribution  k = 2, theta = 2.5
 

    scaler1 = 0.16 * (d1) * torch.exp(- (0.4*d1))

    distances1 = scaler1 * d1

    scaler2 = 0.16 * (d2) * torch.exp(- (0.4*d2))

    distances2 = scaler2 * d2

    # distances = torch.mean(distances)
    # distances = torch.sum(distances)
    # return distances
# ######wighted_gamma distribution  k = 2, theta = 2.5



# ######wighted_CD_Landau
    scaler1 = torch.exp(-((d1 + torch.exp(-d1))/2))

    # distances1 = scaler1 * d1
    # # distances = scaler * (1 - torch.exp(-distances + 1e-7))
    distances1 = scaler1 * (1 - torch.exp(-d1))


    scaler2 = torch.exp(-((d2 + torch.exp(-d2))/2))


    # distances2 = scaler2 * d2
    # # distances = scaler * (1 - torch.exp(-distances + 1e-7))
    distances2 = scaler2 * (1 - torch.exp(-d2))
# ######wighted_CD_Landau


    
    # return (torch.mean(distances1) + torch.mean(distances2)) / 2
    return (torch.sum(distances1) + torch.sum(distances2)) / 2







def calc_cd_one_side_like_weighted(p1, p2):

    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    dist1 = torch.clamp(dist1, min=1e-9)
    # dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    # d2 = torch.sqrt(dist2)








# ######wighted_extreme_value_sigma1.4_miu_0
    

    
    scaler1 = 0.7143 * torch.exp(-(d1/1.4)) * torch.exp(-torch.exp(-(d1/1.4)))

    distances1 = scaler1 * d1



# ######wighted_extreme_value_sigma1.4_miu_0



# ######wighted_Chi-squared distribution k=3
 

    scaler1 = 0.3989422468 * (torch.sqrt(d1)) * torch.exp(- (d1/2))

    distances1 = scaler1 * d1


# ######wighted_Chi-squared distribution k=3





######1023_Log-logistic alpha=5 beta=2
 
    # Numerator = 0.08 * distances
    # Denominator = (1+ (distances**2) * 0.04)**2
    scaler1 = (0.08 * d1) / ((1+ (d1**2) * 0.04)**2)

    distances1 = scaler1 * d1


    # distances = torch.mean(distances)
    # distances = torch.sum(distances)
    # return distances
######1023_Log-logistic  alpha=5 beta=2





# ######wighted_gamma distribution  k = 2, theta = 2.5
 

    scaler1 = 0.16 * (d1) * torch.exp(- (0.4*d1))

    distances1 = scaler1 * d1


    # return distances
# ######wighted_gamma distribution  k = 2, theta = 2.5



# ######wighted_CD_Landau
    scaler1 = torch.exp(-((d1 + torch.exp(-d1))/2))


    distances1 = scaler1 * (1 - torch.exp(-d1))
# ######wighted_CD_Landau

    # return torch.mean(distances1)
    return torch.sum(distances1)







