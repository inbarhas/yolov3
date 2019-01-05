from yolo import YOLO
from PIL import Image
import os
from matplotlib import pyplot
import numpy as np
from classification_train import predict_class
import logging
from classification_train import get_resnet50


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

                    y1 = max(0, np.floor(y1 + 0.5).astype('int32'))
                    x1 = max(0, np.floor(x1 + 0.5).astype('int32'))
                    y2 = min(image.size[1], np.floor(y2 + 0.5).astype('int32'))
                    x2 = min(image.size[0], np.floor(x2 + 0.5).astype('int32'))

                    output_line += "[{},{},{},{},{}]".format(x1, y1, x2 - x1, y2 - y1, predicted_class)
                    if i < (len(boxes) - 1):
                        output_line += ','
                    else:
                        output_line += '\n'

                of.write(output_line)
                logging.debug(output_line)

                if r_image:
                    # visualization is done by obtaining the image that was drawn in the yolo detection, without class remap - so for each class
                    # needs to add +1 to get actual class
                    pyplot.figure()
                    pyplot.imshow(np.asarray(r_image))
                    pyplot.show()

    yolo.close_session()


from optparse import OptionParser
def main():
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="path")
    parser.add_option("-o", "--out", dest="outf")
    (options, args) = parser.parse_args()

    classificator = {}
    logging.debug("loading classifier : resnet50")
    resnet50 = get_resnet50(num_classes=6, w=None)
    resnet50.load_weights('resnet50_best.h5')

    classificator['resnet50'] = resnet50

    yolo_args = {
        'model_path': 'final_single_cust_loss4_anchs.h5',
        'anchors_path': 'bus_anchors.txt',
        'classes_path': 'bus_classes_single.txt',
    }
    detect_img(YOLO(**yolo_args), imgs_path=options.path, outf=options.outf, cls=classificator, remap=True,
               visualize=True)


if __name__ == "__main__":
    main()
