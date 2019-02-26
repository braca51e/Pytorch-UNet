import torch
import torch.nn.functional as F

from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False, criterion):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_bbox = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_bbox = torch.from_numpy(true_bbox).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_bbox = true_bbox.cuda()

        bbox_pred = net(img)[0]

        bbox_probs_flat = bbox_pred.view(-1)
        true_bbox_flat = true_bbox.view(-1)

        loss = criterion(bbox_probs_flat, true_bbox_flat)
        
        tot += loss.item()

    return tot / i
