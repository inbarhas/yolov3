import numpy as np
import ast
import os
from yolo_video import detect_img
from yolo import YOLO, detect_video
from classification_train import get_resnet50
import logging


def maya_run(myAnnFileName, buses):
	
    annFileNameGT = os.path.join(os.getcwd(),'annotationsTrain.txt')
    writtenAnnsLines = {}
    annFileEstimations = open(myAnnFileName, 'w+')
    annFileGT = open(annFileNameGT, 'r')
    writtenAnnsLines['Ground_Truth'] = (annFileGT.readlines())

    for k, line_ in enumerate(writtenAnnsLines['Ground_Truth']):

        line = line_.replace(' ','')
        imName = line.split(':')[0]
        anns_ = line[line.index(':') + 1:].replace('\n', '')
        anns = ast.literal_eval(anns_)
        if (not isinstance(anns, tuple)):
            anns = [anns]
        corruptAnn = [np.round(np.array(x) + np.random.randint(low = 0, high = 100, size = 5)) for x in anns]
        corruptAnn = [x[:4].tolist() + [anns[i][4]] for i,x in enumerate(corruptAnn)]
        strToWrite = imName + ':'
        if(3 <= k <= 5):
            strToWrite += '\n'
        else:
            for i, ann in enumerate(corruptAnn):
                posStr = [str(x) for x in ann]
                posStr = ','.join(posStr)
                strToWrite += '[' + posStr + ']'
                if (i == int(len(anns)) - 1):
                    strToWrite += '\n'
                else:
                    strToWrite += ','
        annFileEstimations.write(strToWrite)
# End


def run(estimatedAnnFileName, busDir):
    classificator = {}
    logging.debug("loading classifier : resnet50")
    get_resnet50.load_weights(os.path.join('model_data', 'vgg_full_trained_weights_final.h5'))
    #    print("loading classifier : mobilenet")
    #    mobilenet = mobilenet1_get_model(num_classes=4) # TODO change this one moving to final dataset
    #    mobilenet.load_weights(os.path.join('model_data', 'mobilenet_final_weights.h5'))

    classificator['svm'] = cls
    classificator['vgg'] = vgg_double
    #   classificator['mobilenet'] = mobilenet

    yolo_args = {
        'model_path': os.path.join('proj_models', 'final_single_cust_loss4_anchs.h5'),
        'anchors_path': os.path.join('model_data', 'bus_anchors.txt'),
        'classes_path': os.path.join('model_data', 'bus_classes_single.txt'),
    }
    detect_img(YOLO(**yolo_args), imgs_path=busDir, outf=estimatedAnnFileName, cls=classificator, remap=True, visualize=False)
