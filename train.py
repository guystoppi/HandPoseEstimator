import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

from handdataset import load_batch

def iou(pred, lbl, pos_threshold):
    intersect = torch.sum((pred > pos_threshold) & (lbl == 1))
    union = torch.sum((pred > pos_threshold) | (lbl == 1))

    return intersect / float(union)

def train(model, device, train_data, val_data=None, options={}):

    writer = SummaryWriter()

    adam_opt = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

    decayRate = 0.96
    adam_w_lrdecay = torch.optim.lr_scheduler.ExponentialLR(optimizer=adam_opt, gamma=decayRate)

    loss = nn.BCEWithLogitsLoss(reduction='none')

    torch.autograd.set_detect_anomaly(True)

    save_path = os.path.join(options.get("CKPT_DIR", "./ckpts"), "keynet.pth")
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    for e in range(options["MAX_EPOCHS"]):
        for i, sample in enumerate(train_data):

            img, lbl = load_batch(sample[0], sample[1])
            img = img.to(device)
            lbl = lbl.to(device)

            pred = model(img)

            loss_output = loss(pred, lbl)

            # the number of positive cells are vastly outnumbered by
            # the number of negative cells, so we scale their respective losses
            # so the optimizer won't have trash recall
            loss_output[lbl == 1] *= options.get("POS_WEIGHTING", 500)
            loss_output = torch.mean(loss_output)

            loss_output.backward()

            adam_opt.step()
            if (i * options["BATCH_SIZE"] + e * len(train_data) * options["BATCH_SIZE"]) % 50000 == 0:
                adam_w_lrdecay.step()


            if i % options.get("LOG_EVERY", 5) == 0:
                step = i * options["BATCH_SIZE"] + e * len(train_data) * options["BATCH_SIZE"]
                writer.add_scalar("Train/Training Loss", loss_output.item(), step)

        writer.add_scalar("Train/Learning Rate", adam_w_lrdecay.get_lr()[0], e)

        if e % options.get("SAVE_EVERY", 1) == 0:
            torch.save(model.state_dict(), save_path)

        if val_data and e % options.get("EVAL_EVERY", 1) == 0:
            mean_iou = 0
            for val_i, val_sample in enumerate(test_data):
                val_img, val_lbl = load_batch(sample[0], sample[1])
                val_img = val_img.to(device)
                val_lbl = val_lbl.to(device)
                
                pred = torch.sigmoid(model(val_img))
                
                mean_iou += iou(pred, val_lbl, options.get("POS_THRESHOLD", 0.9))

            writer.add_scalar("Validation/Mean IOU", mean_iou / len(test_data), e)