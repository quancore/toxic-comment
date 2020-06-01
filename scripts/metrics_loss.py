import numpy as np
import sklearn
import torch

class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,1]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = sklearn.metrics.roc_auc_score(self.y_true, self.y_pred, labels=np.array([0, 1]))
    
    @property
    def avg(self):
        return self.score
    
class ThresholdMeter(object):
    '''
    Compute the best threshold value for labelling using validation data. The threshold value could be used in test data pseudolabeling.
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0.5,0.5])

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,1]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))

    @property
    def find(self):
        # calculate roc curves
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.y_true, self.y_pred, pos_label=1)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        
        return thresholds[ix]
                
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

class FocalLosswSmooting(torch.nn.Module):
    def __init__(self, alpha=0.2, gamma=2., smoothing = 0.1):
        super(FocalLosswSmooting, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
    
    def forward(self, output, target):
        if self.training:
            output = output.float()
            target = target.float()
            output = torch.sigmoid(output)
            # print(f'target_1 {target}')
            # print(f'output: {output}')
            pos_mask = (target == 1).float()
            neg_mask = (target == 0).float()
            # print(f'pos_mask: {pos_mask}')
            # print(f'neg_mask: {neg_mask}')
            pos_loss = -self.alpha * torch.pow(torch.sub(1.0, output), self.gamma) * torch.log(output) * pos_mask
            neg_loss = -(1 - self.alpha) * torch.pow(output, self.gamma) * \
                       torch.log(torch.sub(1.0, output)) * neg_mask
            # print(f'pos_loss {pos_loss}')
            # print(f'neg_loss {neg_loss}')
    
            neg_loss = neg_loss.sum()
            pos_loss = pos_loss.sum()
            num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
            num_neg = neg_mask.view(neg_mask.size(0), -1).sum()
    
            if num_pos == 0:
                loss = neg_loss
            else:
                loss = pos_loss / num_pos + neg_loss / num_neg
            
            # print(loss)
            # print(-output.sum(dim=-1).mean())
            return (1-self.smoothing)*loss + self.smoothing * -output.sum(dim=-1).mean()
        
        else:
            return torch.nn.functional.cross_entropy(output, target)