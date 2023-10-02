import os
import datetime

import torch
import torchvision

from ..network_files import FasterRCNN, AnchorsGenerator
from ..backbone import MobileNetV2
from ..dataloader.iit_dataset import IITDataSet
from ..dataloader.umd_dataset import UMDDataSet
from ..train_utils import GroupedBatchSampler, create_aspect_ratio_groups, transforms
from ..train_utils import train_eval_utils as utils


def create_model(num_classes):

    # https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    backbone = MobileNetV2(weights_path="../backbone/mobilenet_v2.pth").features
    backbone.out_channels = 1280

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=[7, 7],
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if not os.path.exists("../save_weights"):
        os.makedirs("../save_weights")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    root = args.data_path   # dataset root directory


    if os.path.exists(os.path.join(root, "train")) is False:
        raise FileNotFoundError("dataset dose not in path:'{}'.".format(root))

    if args.dataset == 'iit':
        # load train data set
        train_dataset = IITDataSet(root, data_transform["train"], "train")
        # load validation data set
        val_dataset = IITDataSet(root, data_transform["val"], 'val')
    elif args.dataset == 'umd':
        # load train data set
        train_dataset = UMDDataSet(root, data_transform["train"], "train")
        # load validation data set
        val_dataset = UMDDataSet(root, data_transform["val"], 'val')
    train_sampler = None

    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataset workers' % nw)

    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)

    # create  num_classes equal background + 20 classes
    model = create_model(num_classes=args.num_classes + 1)


    model.to(device)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    train_loss = []
    learning_rate = []
    val_map = []

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  first frozen backbone and train 5 epochs                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for param in model.backbone.parameters():
        param.requires_grad = False

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    init_epochs = 5
    save_path = args.output_dir / args.dataset
    for epoch in range(init_epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")
        val_map.append(coco_info[1])  # pascal mAP

    torch.save(model.state_dict(), save_path / "mobile-model.pth")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  second unfrozen backbone and train all network     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    for name, parameter in model.backbone.named_parameters():
        split_name = name.split(".")[0]
        if split_name in ["0", "1", "2", "3"]:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)
    num_epochs = 20
    for epoch in range(init_epochs, num_epochs+init_epochs, 1):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        if epoch in range(num_epochs+init_epochs)[-5:]:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, save_path / "mobile-model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from ..train_utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from ..train_utils.plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--device', default='cuda:0', help='device')

    parser.add_argument('--data_path', default=r'', help='dataset root directory')
    parser.add_argument("--dataset", type=str, default='', choices=['iit', 'umd'],
                        help='Name of dataset')
    parser.add_argument('--num_classes', default=10, type=int, help='num_classes(No background included)')
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--output_dir', default='../save_weights/', help='path where to save')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    main(args)
