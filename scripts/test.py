import os
import json
import torch
import warnings
import pandas as pd
from torch.autograd import Variable
from config.config import parse_args
from scripts.utils import check_paths, prepare_model, prepare_image
warnings.filterwarnings("ignore")


def get_prediction(args, image_path: str, load_model_name: str) -> str:
    """
    :param Image_path: Path to image folder
    :return: Name of the flower
    """
    # CHECK PATHS
    result_path, model_path, train_log_path, val_log_path = check_paths(args)

    # CUDA LAUNCH
    device = torch.device("cuda:%s" % (str(args.gpu)) if torch.cuda.is_available() else "cpu")

    # OPEN CLASSIFICATION DICTIONARY
    json_file = open(args.classes_json)
    idx_to_class = json.load(json_file)
    num_classes = len(idx_to_class)

    # LOAD MODEL
    model = prepare_model(args.model_name, num_classes, model_path)
    load_net = torch.load(os.path.join(model_path, load_model_name))
    model.load_state_dict(load_net)
    model.eval()

    # IMAGE PREPARATION FUNCTION
    image = prepare_image(image_path)

    # IMAGE PREDICTION
    with torch.no_grad():
        input = Variable(image)
        prediction = model(input)

    class_name = idx_to_class[str(int(torch.argmax(prediction[0]).item()) + 1)]

    return class_name

if __name__ == '__main__':
    # PARSE ARGUMENTS
    args = parse_args()

    results = []
    for path in os.listdir(os.path.join(args.data_path, 'test')):
        image_path = (os.path.join(*[args.data_path, 'test', path]))
        class_name = get_prediction(args, image_path, args.trained_model)
        results.append([image_path, class_name])

    data = pd.DataFrame(results, columns=['img_path', 'fl_class'])
    save_path = os.path.join(*[args.experiment_path, args.result_dir_name, f'predictions_{args.trained_model}.csv'])
    data.to_csv(save_path, index=False)
    print('PREDICTION DONE')