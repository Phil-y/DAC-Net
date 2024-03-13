import numpy as np
import torch
import cv2
from medpy import metric
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import morphology
from sklearn.metrics import roc_curve, auc


class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # if not np.any(x):
        #     x[0][0] = 1.0
        # elif not np.any(y):
        #     y[0][0] = 1.0

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.percentile(distances[indexes], 95))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
            ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()
        if torch.sum(pred) == 0:
            pred[0][0][0][0] = 1
            # print(pred)
            # print(torch.sum(pred))
        # print(pred.shape)
        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
            ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
            ).float()

        # print(right_hd, ' ', left_hd)

        return torch.max(right_hd, left_hd)

hd_metric = HausdorffDistance()

def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        # print('TP=0 now!')
        # print('Epoch: {}'.format(epoch))
        # print('i_batch: {}'.format(i_batch))
        TP = torch.Tensor([1]).cuda()

    #Precision or positive predictive value
    Precision = TP / (TP + FP)

    #recall = Sensitivity
    Recall = TP / (TP + FN)

    # Specificity or true negative rate
    Specificity = TN / (TN + FP)

    #accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    # IoU = Jaccard index
    IoU = TP / (TP + FP + FN)

    # DICE
    DICE = 2 * IoU / (IoU + 1)

    # F1 score = Dice
    F1 = 2 * Precision * Recall / (Precision + Recall)

    #MAE
    MAE = torch.abs(pred - gt).mean()



    # roc
    fpr, tpr, threshold = roc_curve(gt_binary.cpu().numpy().flatten(), pred_binary.cpu().numpy().flatten())
    auc_roc = auc(fpr, tpr)

    #hd
    # hd = hd_metric.compute(pred, gt)


    return Precision, Recall, Specificity, accuracy, IoU, DICE, MAE, auc_roc, F1


class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {}
        for metric in metrics_list:
            self.metrics[metric] = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.metrics.keys()), "The k {} is not in metrics".format(k)
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.metrics[k] += v

    def mean(self, total):
        mean_metrics = {}
        for k, v in self.metrics.items():
            mean_metrics[k] = v / total
        return mean_metrics


# if __name__ == '__main__':
#     pred = torch.sigmoid(torch.randn(1, 1, 224, 224))
#     target = torch.sigmoid(torch.randn(1, 1, 224, 224))
#     # _recall, _specificity, _precision, _F1, _F2, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean, MAE, DICE, HD = evaluate(pred, target)
#     Precision, Recall, Specificity, accuracy, IoU, DICE, F1, MAE, auc_roc, hd = evaluate(pred, target)
#     # print(torch.abs(pred - target).mean())
#     print(evaluate(pred, target))
