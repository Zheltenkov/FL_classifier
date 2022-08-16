import os
import json
import time
import torch
import warnings
from config.config import parse_args
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from utils import check_paths, prepare_model
from train_one_epoch import train_loop, valid_loop
from dataloader.dataloader import get_dataloader
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # PARSE ARGUMENTS
    args = parse_args()

    # CHECK PATHS
    result_path, model_path, train_log_path, val_log_path = check_paths(args)

    # CUDA LAUNCH
    device = torch.device("cuda:%s" % (str(args.gpu)) if torch.cuda.is_available() else "cpu")
    print('----------WORKING DEVICE IS ' + str(str(device).split(":")[0].upper()) + ' ----------')


    json_file = open(args.classes_json)
    idx_to_class = json.load(json_file)
    num_classes = len(idx_to_class)

    # INITIALIZED DATA LOADERS
    dataloader = get_dataloader(args, idx_to_class)
    print('----------PREPARE DATALOADERS ----------')

    # LOAD MODEL
    model = prepare_model(args.model_name, num_classes, model_path)

    # FREEZE WEIGHTING COEFFICIENTS OF THE MODEL ON ALL LAYERS EXCEPT THE OUTPUT AND TRAIN THE MODEL
    for param in model.parameters():
        param.require_grad = True
    print('----------LOAD MODELS----------')

    # SEND MODEL ON GPU
    model.to(device)
    print('----------SEND MODELS ON GPU ----------')

    # PREPARE LOSS FUNCTION
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    criterion.to(device)

    # PREPARE OPTIMIZER
    param_list = (list(model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    #scheduler_caer = StepLR(optimizer, step_size=5, gamma=0.1)
    print('---------PREPARE OPTIMIZER ----------')



    print('----------START TRAINING ----------')
    # SET START TIME AND WRITERS
    since = time.time()

    train_writer = SummaryWriter(os.path.join(train_log_path, "model_train"))
    valid_writer = SummaryWriter(os.path.join(val_log_path, "model_valid"))

    for epoch_n in range(args.epochs):
        # CALL TRAIN/VALID FUNCTIONS FOR EPOCH
        report_train = open('../debug_exp/results/report_train.txt', 'a')
        loss_ls_tr, acc_ls_tr, f1_score_tr = train_loop(epoch_n, model, dataloader['train'], criterion,
                                                        optimizer, device, train_writer, num_classes)
        report_train.writelines(f'{loss_ls_tr},{acc_ls_tr},{f1_score_tr} \n')
        report_train.close()

        report_valid = open('./debug_exp/results/report_valid.txt', 'a')
        loss_ls_val, acc_ls_val, f1_score_val = valid_loop(epoch_n, model, dataloader['valid'], criterion,
                                                           device, valid_writer, num_classes)
        report_valid.writelines(f'{loss_ls_val}, {acc_ls_val}, {f1_score_val} \n')
        report_valid.close()

        if epoch_n % 2 == 0:
            print("--------------CHECK POINT-------------------------")
            torch.save(model.state_dict(), os.path.join(model_path, '{0}_model_{1}.pth'.format(args.model_name, epoch_n)))

    time_elapsed = time.time() - since
    print('Training time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))