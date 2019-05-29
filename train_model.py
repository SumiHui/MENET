# -*- coding: utf-8 -*-
# @File    : derain_pytorch/train_model.py
# @Info    : @ TSMC-SIGGRAPH, 2019/4/16
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os

import numpy as np
import torch.autograd
import torch.optim as optim

from configuration import cfg
from data_helper_h5 import get_batch
from networks import DerainNet, RegressionTrain

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    n_tasks = 2
    trainer = RegressionTrain(DerainNet(), n_tasks)

    optimizer = optim.SGD(params=trainer.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    weights = []
    task_losses = []
    loss_ratios = []
    grad_norm_losses = []

    for epoch in range(cfg.epochs):
        print("--------{}--------".format(epoch))
        torch.cuda.empty_cache()
        dataloader = get_batch("RainTrainL.h5", cfg.batch_size)
        for idx, sample in enumerate(dataloader):
            # print("input tensor shape: ", sample['input'].size())
            trainer.train(mode=True)
            # trainer.model.zero_grad()
            # optimizer.zero_grad()

            # bs, ncrops, c, h, w = sample['input'].size()
            # x = sample['input'].view(-1, c, h, w)
            # y = sample['label'].view(-1, c, h, w)
            x, y = sample
            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)
            # evaluate each task loss L_i(t)
            task_loss = trainer(x, y)  # this will do a forward pass in the model and will also evaluate the loss
            # compute the weighted loss w_i(t) * L_i(t)
            weighted_task_loss = torch.mul(trainer.weights, task_loss)
            # initialize the initial loss L(0) if i=0
            if idx == 0:
                # set L(0)
                if torch.cuda.is_available():
                    initial_task_loss = task_loss.data.cpu()
                else:
                    initial_task_loss = task_loss.data
                # initial_task_loss = initial_task_loss.numpy()

            # get the total loss
            loss = torch.sum(weighted_task_loss)
            # print('|--- raw total loss:  {}'.format(loss))
            # clear the gradients
            optimizer.zero_grad()
            # do the backward pass to compute the gradients for the whole set of weights
            # This is equivalent to compute each \n abla_W L_i(t)
            loss.backward(retain_graph=True)
            # print("|--- after backward, tasks grad: {}".format(trainer.weights.grad))
            # trainer.weights.grad.data = trainer.weights.grad.data * 0.0

            if cfg.mode == "grad_norm":
                # get layer of shared weights
                W = trainer.get_last_shared_layer()
                # get the gradient norms for each of the tasks G^{(i)}_w(t)
                norms = []
                # for i in range(len(task_loss)):
                for i in range(2):
                    # get the gradient of this task loss with respect to the shared parameters
                    gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                    # compute the norm, fixme:(bug) IndexError: tuple index out of range
                    # print("     |--- trainer.weights[{}]:  {}".format(i, trainer.weights[i]))
                    norms.append(torch.norm(torch.mul(trainer.weights[i], gygw[0])))
                norms = torch.stack(norms)
                # print("|--- norms {}".format(norms))

                if torch.cuda.is_available():
                    loss_ratio = task_loss.data.cpu() / initial_task_loss
                else:
                    loss_ratio = task_loss.data / initial_task_loss
                # r_i(t)
                inverse_train_rate = loss_ratio / torch.mean(loss_ratio)
                # print("|--- inverse_train_rate: {}".format(inverse_train_rate))

                if torch.cuda.is_available():
                    mean_norm = torch.mean(norms.data.cpu())
                else:
                    mean_norm = torch.mean(norms.data)

                constant_term = mean_norm * (inverse_train_rate ** cfg.alpha)
                constant_term = constant_term.detach()
                # print("|--- mean_norm: {}, constant_term: {}".format(mean_norm, constant_term))
                if torch.cuda.is_available():
                    constant_term = constant_term.cuda()

                # grad_norm_loss = torch.tensor(torch.sum(torch.abs(norms - constant_term)))
                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                # print('|--- GradNorm loss:  {}'.format(grad_norm_loss))

                # compute the gradient for the weights
                # print("|--- trainer.weights.grad: {}".format(trainer.weights.grad))
                new_grad = torch.autograd.grad(grad_norm_loss, trainer.weights)[0]
                # print("==>", new_grad)
                trainer.weights.grad.data = new_grad.data
                # print("==>", trainer.weights.grad.data)

            optimizer.step()
            # if idx % 10 == 0:
            #     print("iter {}: loss {}".format(idx, loss.data))

            # renormalize
            normalize_coeff = n_tasks / torch.sum(trainer.weights.data, dim=0)
            trainer.weights.data = trainer.weights.data * normalize_coeff

            # record
            if torch.cuda.is_available():
                task_losses.append(task_loss.data.cpu().numpy())
                loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
                weights.append(trainer.weights.data.cpu().numpy())
                grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
            else:
                task_losses.append(task_loss.data.numpy())
                loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
                weights.append(trainer.weights.data.numpy())
                grad_norm_losses.append(grad_norm_loss.data.numpy())

            if idx % 20 == 0:
                if torch.cuda.is_available():
                    print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
                        epoch, idx, loss_ratios[-1], trainer.weights.data.cpu().numpy(), task_loss.data.cpu().numpy(),
                        grad_norm_loss.data.cpu().numpy()))
                else:
                    print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
                        epoch, idx, loss_ratios[-1], trainer.weights.data.numpy(), task_loss.data.numpy(),
                        grad_norm_loss.data.numpy()))

    # the end of n epochs
    trainer.model.eval()
    checkpoint = {
        'state_dict': trainer.model.state_dict(),
        'opt_state_dict': optimizer.state_dict(),
        'epoch': cfg.epochs
    }
    if not os.path.exists("ckpt"):
        os.mkdir("ckpt")
    torch.save(checkpoint, os.path.join("ckpt", "checkpoint.data"))

    print("{}".format("testing finish"))

    # draw chart
    task_losses = np.array(task_losses)
    weights = np.array(weights)

    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title('Loss scale $\sigma_0=1.0$)')
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title('Loss (scale $\sigma_1=?)$')
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("$\sum_i L_i(t) / L_i(0)$")
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title('$L_{grad}$')

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title('Change of weights $w_i$ over time')

    ax1.plot(task_losses[:, 0])
    ax2.plot(task_losses[:, 1])
    ax3.plot(loss_ratios)
    ax4.plot(grad_norm_losses)
    ax5.plot(weights[:, 0])
    ax5.plot(weights[:, 1])
    # plt.show()
    plt.savefig("results.jpg")
