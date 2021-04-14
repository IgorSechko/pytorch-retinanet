import argparse
import collections
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from retinanet import coco_eval
from retinanet import csv_eval_new_metric
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

CSV_TRAIN = "/home/ITRANSITION.CORP/i.sechko/datasets/Jacket_Tshirt/csvs/jacket_t-shirt_1000_1000.csv"
CSV_CLASSES = '/home/ITRANSITION.CORP/i.sechko/datasets/Jacket_Tshirt/csvs/jacket_t-shirt_classes.csv'
CSV_VAL = "/home/ITRANSITION.CORP/i.sechko/datasets/Jacket_Tshirt/csvs/jacket_t-shirt_1000_1000_val_100_100.csv"
DATASET = "csv"
RESNET_DEPTH = 50  # 18,34,50,101,152

MODEL_FINAL = '/home/ITRANSITION.CORP/i.sechko/datasets/Jacket_Tshirt/models_1000/model_final.pt'
MODEL_EPOCH = '/home/ITRANSITION.CORP/i.sechko/datasets/Jacket_Tshirt/models_1000/1_{}_retinanet_{}.pt'
RESULTS = "/home/ITRANSITION.CORP/i.sechko/datasets/Jacket_Tshirt/results/results_1000.txt"
RUN_NAME = "coco_1000_100_1l"

LOAD_PRETRAINED = True
PRETRAINED_MODEL = "/home/ITRANSITION.CORP/i.sechko/datasets/Jacket_Tshirt/models_1000/1_csv_retinanet_26.pt"
DEL_LAST_LAYERS = False

START_EPOCH = 27
END_EPOCH = 100


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default=DATASET)
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)',
                        default=CSV_TRAIN)
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)', default=CSV_CLASSES)
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)',
                        default=CSV_VAL)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int,
                        default=RESNET_DEPTH)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=END_EPOCH)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        # image_dir = '/home/ITRANSITION.CORP/v.shamkina/data/Fashion_Dataset/train/Jacket_Dress_Jeans/'

        # data = AugmenterService(parser.csv_train, parser.csv_classes, image_dir)
        # data.augmentation()

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Resizer()])) #Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=0, collate_fn=collater, batch_sampler=sampler)

    dataloader_val = None
    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
        if LOAD_PRETRAINED:
            state_dict = torch.load(PRETRAINED_MODEL)
            if DEL_LAST_LAYERS:
                del state_dict['classificationModel.output.weight']
                del state_dict['classificationModel.output.bias']
                del state_dict['regressionModel.output.weight']
                del state_dict['regressionModel.output.bias']
            retinanet.load_state_dict(state_dict=state_dict, strict=False)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    for param in retinanet.parameters():
        param.requires_grad = False

    retinanet.regressionModel.output.weight.requires_grad = True
    retinanet.regressionModel.output.bias.requires_grad = True

    retinanet.classificationModel.output.weight.requires_grad = True
    retinanet.classificationModel.output.bias.requires_grad = True

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1e-2, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    writer = SummaryWriter(os.environ["HOME"] + "/tensorboard/runs/" + RUN_NAME)

    for epoch_num in range(START_EPOCH, parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        epoch_classification_loss = []
        epoch_regression_loss = []

        print("epoch:", epoch_num)

        for iter_num, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                epoch_classification_loss.append(float(classification_loss))
                epoch_regression_loss.append(float(regression_loss))

                # print(
                #     'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                #         epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        epoch_loss = np.mean(epoch_loss)
        epoch_classification_loss = np.mean(epoch_classification_loss)
        epoch_regression_loss = np.mean(epoch_regression_loss)

        writer.add_scalar("Train Loss/classification+regression", epoch_loss, epoch_num)
        writer.add_scalar("Train Loss/classification", epoch_classification_loss, epoch_num)
        writer.add_scalar("Train Loss/regression", epoch_regression_loss, epoch_num)

        print("Evaluating train set...")
        mAP, new_metric = csv_eval_new_metric.evaluate(dataset_train, retinanet)

        tag_scalar = {f"class {key}": val[0] for key, val in mAP.items()}
        writer.add_scalars("Train mAP", tag_scalar, epoch_num)

        tag_scalar = {f"class {key}": val for key, val in new_metric.items()}
        writer.add_scalars("Train Percent match annotations", tag_scalar, epoch_num)

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            if dataloader_val is not None:
                print("Calculating validation loss...")
                val_loss, val_classification_loss, val_regression_loss = get_val_loss(retinanet, dataloader_val)

                writer.add_scalar("Validation Loss/classification+regression", val_loss, epoch_num)
                writer.add_scalar("Validation Loss/classification", val_classification_loss, epoch_num)
                writer.add_scalar("Validation Loss/regression", val_regression_loss, epoch_num)

            print('Evaluating validation set...')
            mAP, new_metric = csv_eval_new_metric.evaluate(dataset_val, retinanet)

            tag_scalar = {f"class {key}": val[0] for key, val in mAP.items()}
            writer.add_scalars("Validation mAP", tag_scalar, epoch_num)

            tag_scalar = {f"class {key}": val for key, val in new_metric.items()}
            writer.add_scalars("Val Percent match annotations", tag_scalar, epoch_num)

            with open(RESULTS, "a") as outfile:
                outfile.write('Number of epoch: ' + str(epoch_num))
                for key, val in mAP.items():
                    outfile.write('\n{}:{}\n'.format(key, val))

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module.state_dict(), MODEL_EPOCH.format(parser.dataset, epoch_num))

    retinanet.eval()
    torch.save(retinanet.state_dict(), MODEL_FINAL)


def get_val_loss(retinanet, dataloader_val):
    val_loss = []
    val_classification_loss = []
    val_regression_loss = []

    retinanet.train()
    retinanet.module.freeze_bn()
    with torch.no_grad():

        for iter_num, data in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):
            try:

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                val_loss.append(float(loss))
                val_classification_loss.append(float(classification_loss))
                val_regression_loss.append(float(regression_loss))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

    return np.mean(val_loss), np.mean(val_classification_loss), np.mean(val_regression_loss)


if __name__ == '__main__':
    main()
