def dice(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()

    tp = (pred * target).sum()
    total = pred.sum() + target.sum()

    return (2 * tp + eps)/ (total + eps)

def iou(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()

    tp = (pred * target).sum()
    union = pred.sum() + target.sum() - tp

    return (tp + eps) / (union + eps)