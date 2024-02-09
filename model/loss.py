import torch


def iou(pred, labels,average=True,return_num=False):
    e = 1e-6
    #pred = torch.sigmoid(pred)
    pred = (pred>0.5).float()
    labels = (labels>0.5).float()
    intersection = pred * labels
    union = (pred + labels) - intersection
    iou = intersection.sum(-1).sum(-1).sum(-1) / (union.sum(-1).sum(-1).sum(-1) + e)
    if return_num:
        num = (iou!=1.).sum()
        return iou[iou!=1].sum(), num

    if average:
        return iou.mean()
    else:
        return iou.sum()

def focal_loss(pred,gt,gamma=2,average=False,loss_mask=None):
    #pred = torch.sigmoid(pred)
    probs = pred * gt + (1 - pred) * (1 - gt)
    if loss_mask is None:
        batch_loss = -(torch.pow((1-probs), gamma))*torch.log(probs+1e-06)
    else:
        batch_loss = -(torch.pow((1-probs), gamma))*torch.log(probs+1e-06)*loss_mask
    if average:
        return batch_loss.mean()
    else:
        return batch_loss.sum()

if __name__ == '__main__':
    x = torch.sigmoid(torch.rand(1,1,1,16,16))
    y = torch.ones(1,1,1,16,16)
    print(focal_loss(x,y,2))