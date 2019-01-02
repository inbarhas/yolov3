import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
from matplotlib import pyplot
import numpy as np
from sklearn.externals import joblib
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from keras.models import Model
from classification_train import predict_class, mobilenet1_get_model, vgg16_get_model
import logging

classes_remap = {
    0: 1, # Green
    1: 2, # Yellow
    2: 3, # White
    3: 4, # Silver/Grey
    4: 5, # Blue
    5: 6, # Red
}


def detect_img(yolo, imgs_path, outf, cls=None, remap=False, visualize=False):
    with open(outf, 'w') as of:
        for imgname in os.listdir(imgs_path):
            if imgname.lower().endswith('.jpg') or imgname.lower().endswith('.jpeg'):
                logging.debug('Input image filename:{}'.format(imgname))
                image = Image.open(os.path.join(imgs_path, imgname))
                r_image, boxes, scores, classes = yolo.detect_image(image, predict_class, cls, visualize=visualize)
                """
                # Classify this bounding box
                if cls:
                    # boxes is a list of [top, left, bottom, right] i.e [y1, x1, y2, x2]
                    y = predict_class(image, boxes, cls)
                    print('y = {}'.format(y))
                """

                if len(boxes) == 0:
                    continue

                # Output format
                # PIC.JPG:[xmin1,ymin1,width1,height1,color1],..,[xminN,yminN,widthN,heightN,colorN]
                output_line = "{}:".format(imgname)
                for i, b in enumerate(boxes):
                    y1, x1, y2, x2 = b
                    predicted_class = classes[i]
                    if remap:
                        predicted_class = classes_remap[int(predicted_class)]

                    output_line += "[{},{},{},{},{}]".format(x1, y1, x2 - x1, y2 - y1, predicted_class)
                    if i < (len(boxes) - 1):
                        output_line += ','
                    else:
                        output_line += '\n'

                of.write(output_line)
                logging.debug(output_line)

                if r_image:
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
    parser.add_argument('--path', type=str, help='test path')
    FLAGS = parser.parse_args()

    classificator = {}
    print("loading classifier : svm / vgg")
    cls = joblib.load('model_data/svm.dump')
    vgg_double, _ = vgg16_get_model(num_classes=4)
    print("loading vgg weights")
    vgg_double.load_weights('model_data/vgg_full_trained_weights_final.h5')
#    print("loading classifier : mobilenet")
#    mobilenet = mobilenet1_get_model(num_classes=4) # TODO change this one moving to final dataset
#    mobilenet.load_weights('model_data/mobilenet_final_weights.h5')

    classificator['svm'] = cls
    classificator['vgg'] = vgg_double
 #   classificator['mobilenet'] = mobilenet

    """
    Image detection mode, disregard any remaining command line arguments
    """
    print("Image detection mode")
    if "input" in FLAGS:
        print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)

    detect_img(YOLO(**vars(FLAGS)), imgs_path=FLAGS.path, outf='result.txt', remap=False, cls=classificator, visualize=True)

    """
    yolo_args = {
        'model_path': os.path.join('proj_models', 'final_single_cust_loss4_anchs.h5'),
        'anchors_path': os.path.join('model_data', 'bus_anchors.txt'),
        'classes_path': os.path.join('model_data', 'bus_classes_single.txt'),
    }
    detect_img(YOLO(**yolo_args), FLAGS.path, classificator, visualize=True)
    """
