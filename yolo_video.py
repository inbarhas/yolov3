import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
from matplotlib import pyplot
import numpy as np


def detect_img(yolo, path):
    for imgname in os.listdir(path):
        if imgname.lower().endswith('.jpg') or imgname.lower().endswith('.jpeg'):
            print('Input image filename:{}'.format(imgname))
            image = Image.open(os.path.join(path, imgname))
            r_image = yolo.detect_image(image)
            pyplot.figure()
            pyplot.imshow(np.asarray(r_image))
            pyplot.show()
    yolo.close_session()


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str, dest='model_path',
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )
    parser.add_argument(
        '--anchors', type=str,dest='anchors_path',
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )
    parser.add_argument(
        '--classes', type=str, dest='classes_path',
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )
    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument('--path', type=str, help='test path')
    FLAGS = parser.parse_args()
    FLAGS.image = True

    """
    Image detection mode, disregard any remaining command line arguments
    """
    print("Image detection mode")
    if "input" in FLAGS:
        print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
    detect_img(YOLO(**vars(FLAGS)), FLAGS.path)
