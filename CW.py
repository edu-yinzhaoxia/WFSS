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

    save_path_prefix = args.save_path_prefix + 'Exp_' + DataName[args.dataID] + '/'

    if os.path.exists(save_path_prefix) == False:
        os.makedirs(save_path_prefix)

    num_epochs = 500
    if args.model == 'WFSS':
        Model = WFSS(num_features=num_features, num_classes=num_classes, dim=dim)



    Model = Model.cuda()
    Model.train()
    optimizer = torch.optim.Adam(Model.parameters(), lr=args.lr, weight_decay=args.decay)

    images = torch.from_numpy(X_train).float().cuda()
    label = torch.from_numpy(Y_train).long().cuda()
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
    output = Model(images)
    _, predict_labels = torch.max(output, 1)
    predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)
    OA_clean, kappa_clean, ProducerA_clean = CalAccuracy(predict_labels[test_array], Y[test_array])
    print('OA_clean=%.3f,Kappa_clean=%.3f' % (OA_clean * 100, kappa_clean * 100))
    print('producerA_clean:', ProducerA_clean)

    # CW attack
    def singleAttack(Model, processed_image, label_mask, C=50, iter_eps=0.04):
        processed_image = Variable(processed_image).cuda()
        processed_image = processed_image.requires_grad_()
        output = Model(processed_image)
        correct_logit = torch.mean(torch.sum(label_mask * output, dim=1))
        wrong_logit = torch.mean(torch.max((1 - label_mask) * output, dim=1)[0])
        loss = correct_logit - wrong_logit + C
        Model.zero_grad()
        loss.backward()
        grad = processed_image.grad.data.cpu()
        pertubation = iter_eps * grad / torch.norm(grad, float("inf"))
        processed_image = processed_image.detach().cpu() + pertubation
        processed_image = torch.clamp(processed_image, 0, 1)
        return processed_image

    nb_iter = 10

    processed_image = images.cpu()
    label = torch.from_numpy(Y_tar).long().cuda()
    label_mask = F.one_hot(label, num_classes=num_classes).cuda()
    label_mask = torch.moveaxis(label_mask, -1, 1)

    for i in range(nb_iter):
        processed_image = singleAttack(Model, processed_image, label_mask)

    X_adv = torch.clamp(processed_image, 0, 1).data.numpy()[0]
    X_adv = np.reshape(X_adv, (1, num_features, h, w))
    adv_images = torch.from_numpy(X_adv).float().cuda()

    output = Model(adv_images)
    _, predict_labels = torch.max(output, 1)
    predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)

    OA_cw, kappa_cw, ProducerA_cw = CalAccuracy(predict_labels[test_array], Y[test_array])

    print('OA_cw=%.3f,Kappa_cw=%.3f' % (OA_cw * 100, kappa_cw * 100))
    print('producerA_cw:', ProducerA_cw)

    return OA_clean, OA_cw, ProducerA_cw, kappa_cw


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


