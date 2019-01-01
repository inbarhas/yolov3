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


def detect_img(yolo, path, cls=None, cls_type='svm'):
    for imgname in os.listdir(path):
        if imgname.lower().endswith('.jpg') or imgname.lower().endswith('.jpeg'):
            print('Input image filename:{}'.format(imgname))
            image = Image.open(os.path.join(path, imgname))
            r_image, boxes, scores = yolo.detect_image(image, predict_class, cls)
            """
            # Classify this bounding box
            if cls:
                # boxes is a list of [top, left, bottom, right] i.e [y1, x1, y2, x2]
                y = predict_class(image, boxes, cls)
                print('y = {}'.format(y))
            """
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
#    parser.add_argument(
#        '--post', type=str, dest='post_cls', default='svm',
#        help='post classificator to use, default is SVM',
#    )

    parser.add_argument('--path', type=str, help='test path')
    FLAGS = parser.parse_args()
    FLAGS.image = True

    classificator = {}
    cls_type = 'svm'
    print("loading classifier : svm / vgg")
    cls = joblib.load('model_data/svm.dump')
    vgg_classifier, vgg_feature_extractor = vgg16_get_model(num_classes=4)
    print("loading vgg weights")
    vgg_classifier.load_weights('model_data/vgg_full_trained_weights_final.h5')
    print("loading classifier : mobilenet")
    mobilenet = mobilenet1_get_model(num_classes=4) # TODO change this one moving to final dataset
    mobilenet.load_weights('model_data/mobilenet_final_weights.h5')

    classificator['svm'] = cls
    classificator['vgg_features'] = vgg_feature_extractor
    classificator['vgg_classifier'] = vgg_classifier
    classificator['mobilenet'] = mobilenet

    """
    Image detection mode, disregard any remaining command line arguments
    """
    print("Image detection mode")
    if "input" in FLAGS:
        print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
    detect_img(YOLO(**vars(FLAGS)), FLAGS.path, classificator, cls_type)
