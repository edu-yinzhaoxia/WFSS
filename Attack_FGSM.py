import os
import time
import argparse
import torch
from torch.autograd import Variable
from HyperTools import *
# from Models import *
from WFSS import *
from tqdm import tqdm

DataName = {1: 'PaviaU', 2: 'Salinas', 3: 'Indian_pines'}


def main(args):
    if args.dataID == 1:
        num_classes = 9
        num_features = 103
        # num_tokens = 148*80
        dim = 148 * 80
        save_pre_dir = './Data/PaviaU/'
    elif args.dataID == 2:
        num_classes = 16
        num_features = 204
        # num_tokens = 123*49
        dim = 123 * 49
        save_pre_dir = './Data/Salinas/'
    elif args.dataID == 3:
        num_classes = 16
        num_features = 200
        # num_tokens = 31*31
        dim = 31 * 31
        save_pre_dir = './Data/Indian_pines/'

    X = np.load(save_pre_dir + 'X.npy')
    _, h, w = X.shape
    Y = np.load(save_pre_dir + 'Y.npy')

    X_train = np.reshape(X, (1, num_features, h, w))
    train_array = np.load(save_pre_dir + 'train_array.npy')
    test_array = np.load(save_pre_dir + 'test_array.npy')
    Y_train = np.ones(Y.shape) * 255
    Y_train[train_array] = Y[train_array]
    Y_train = np.reshape(Y_train, (1, h, w))

    # define the targeted label in the attack
    Y_tar = np.zeros(Y.shape)
    Y_tar = np.reshape(Y_tar, (1, h, w))


    num_epochs = 500
    if args.model == 'WFSS':
        Model = WFSS(num_features=num_features, num_classes=num_classes, dim=dim)

    Model = Model.cuda()
    images = torch.from_numpy(X_train).float().cuda()
    label = torch.from_numpy(Y_train).long().cuda()
    Model.train()
    optimizer = torch.optim.Adam(Model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = CrossEntropy2d().cuda()

    # for epoch in range(num_epochs):
    for epoch in tqdm(range(num_epochs)):
        adjust_learning_rate(optimizer, args.lr, epoch, num_epochs)
        tem_time = time.time()
        optimizer.zero_grad()
        output = Model(images)

        seg_loss = criterion(output, label)
        seg_loss.backward()

        optimizer.step()

        batch_time = time.time() - tem_time
        # if (epoch+1) % 1 == 0:
        #     print('epoch %d/%d:  time: %.2f cls_loss = %.3f'%(epoch+1, num_epochs,batch_time,seg_loss.item()))
        time.sleep(0.1)


    Model.eval()

    # clean test
    output_clean = Model(images)
    _, predict_labels = torch.max(output_clean, 1)
    predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)
    OA_clean, kappa_clean, ProducerA_clean = CalAccuracy(predict_labels[test_array], Y[test_array])
    print('OA_clean=%.3f,Kappa_clean=%.3f' % (OA_clean * 100, kappa_clean * 100))
    print('producerA_clean:', ProducerA_clean)

    # FGSM
    image = Variable(images)
    image = image.requires_grad_()
    label = torch.from_numpy(Y_tar).long().cuda()

    output = Model(image)
    seg_loss = criterion(output, label)
    seg_loss.backward()
    adv_noise = args.epsilon * image.grad.data / torch.norm(image.grad.data, float("inf"))

    image.data = image.data - adv_noise
    X = torch.clamp(image, 0, 1).cpu().data.numpy()[0]
    X = np.reshape(X, (1, num_features, h, w))

    adv_images = torch.from_numpy(X).float().cuda()

    output = Model(adv_images)
    _, pre = torch.max(output, 1)
    pre = np.squeeze(pre.detach().cpu().numpy()).reshape(-1)
    OA_f, kappa_f, ProducerA_f = CalAccuracy(pre[test_array], Y[test_array])
    print('OA_f=%.3f,Kappa_f=%.3f' % (OA_f * 100, kappa_f * 100))
    print('producerA_f:', ProducerA_f)

    return OA_clean, OA_f, ProducerA_f, kappa_f



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataID', type=int, default=3)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--model', type=str, default='WFSS')

    # train
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--decay', type=float, default=5e-5)
    parser.add_argument('--epsilon', type=float, default=0.04)

    main(parser.parse_args())


