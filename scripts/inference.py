import os
import cv2
from scripts.test import get_prediction
from config.config import parse_args


if __name__ == '__main__':
    # PARSE ARGUMENTS
    args = parse_args()

    # GET IMAGE PATH AND PREDICT CLASS
    images_paths = os.listdir(args.path_input_img)

    for path in images_paths:
        image_path = (os.path.join(args.path_input_img, path))
        class_name = get_prediction(args, image_path, args.trained_model)

        # READ IMAGE
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f'{class_name.upper()}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # SAVE IMAGE
        save_path = (os.path.join(args.path_prediction, image_path.split('\\')[-1]))
        cv2.imwrite(save_path, image)

        # SHOW IMAGE
        cv2.imshow(f'{class_name.upper()}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        os.remove(image_path)