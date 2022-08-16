import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'valid'])
    parser.add_argument('--data_path', type=str, default="./dataset", required=False,
                        help='Path to image directories')
    parser.add_argument('--classes_json', type=str, default="./dataset/cat_to_name.json", required=False,
                        help='Path to json description file classes')
    parser.add_argument('--experiment_path', type=str, required=False, default="debug_exp",
                        help='Path to save experiment files (results, model, logs, reports)')
    parser.add_argument('--model_dir_name', type=str, default='model', help='Name of the directory to save model')
    parser.add_argument('--result_dir_name', type=str, default='results',
                        help='Name of the directory to save results(predictions, labels mat files)')
    parser.add_argument('--log_dir_name', type=str, default='logs',
                        help='Name of the directory to save logs (train, valid)')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_models', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='alexnet', help='Architecture of pretrained model')
    parser.add_argument('--trained_model', type=str, default='alexnet_model_28.pth', help='Trained model')
    parser.add_argument('--path_input_img', type=str, default='./data/input_data', help='Path of input single image')
    parser.add_argument('--path_prediction', type=str, default='./data/prediction', help='Path of output single image')

    args = parser.parse_args()

    return args