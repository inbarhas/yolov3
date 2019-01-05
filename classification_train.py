import numpy as np
import PIL
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import logging
import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

classes_list = []

network_input_shape = {
    'resnet50': (224, 224),
}


def process_lines(lines, net_type='resnet50'):
    """
    lines shape
    /home/tamirmal/workspace/git/tau_proj_prep/TRAIN/IMG_20181226_180908_HHT.jpg 921,1663,1646,2282,0 2066,1459,2698,2002,0 2866,1067,3695,1664,0
    /home/tamirmal/workspace/git/tau_proj_prep/TRAIN/IMG_20181227_001359_HHT.jpg 1717,1431,2721,2151,0
    """
    data = []
    y = []

    visualize = False

    for line in lines:
        img_path = line.split()[0]
        try:
            img = PIL.Image.open(img_path)
            if img is None:
                print("Falied to open {}".format(img_path))
                continue
        except:
            print("Falied to open {}".format(img_path))
            continue

        for box in line.split()[1:]:
            tokens = box.split(',')
            tokens = [int(t) for t in tokens]
            x1, y1, x2, y2 = tokens[0:4]
            gt = tokens[4]
            h = y2 - y1
            w = x2 - x1
            x1, y1, x2, y2 = x1, y1, x1 + w, y1 + h
            # crop : left, upper, right, lower
            copy_im = img.copy()
            cropped = copy_im.crop((x1, y1, x2, y2))
            cropped = cropped.resize(network_input_shape[net_type])
            sample = image.img_to_array(cropped)
            #sample = np.expand_dims(sample, axis=0)

            ########## DEBUG HOOKS ############
            if visualize:
                from matplotlib import pyplot
                pyplot.figure()
                pyplot.imshow(cropped)
                pyplot.show()
            ###################################
            data.append(sample)
            y.append(gt)

            # Now add the image +10% from the sides. (the -10% will be generated from data aug)
            dw = 0.1 * w
            dh = 0.1 * h
            x1 -= dw
            x2 += dw
            y1 -= dh
            y2 += dh

            cols, rows = img.size
            if x1 < 0:
                x1 = 0
            if x2 > cols:
                x2 = cols - 1
            if y1 < 0:
                y1 = 0
            if y2 > rows:
                y2 = rows - 1

            cropped = copy_im.crop((x1, y1, x2, y2))
            cropped = cropped.resize(network_input_shape[net_type])
            sample = image.img_to_array(cropped)
            # sample = np.expand_dims(sample, axis=0)

            ########## DEBUG HOOKS ############
            if visualize:
                from matplotlib import pyplot
                pyplot.figure()
                pyplot.imshow(cropped)
                pyplot.show()
            ###################################
            data.append(sample)
            y.append(gt)

            # Count how much classes we have
            if gt not in classes_list:
                classes_list.append(gt)

    data = np.array(data)
    y = np.array(y)

    hist, bin_edges = np.histogram(y, bins=[0, 1, 2, 3, 4, 5, 6])
    print("hist y:\n{}".format(hist))

    return data, y
# End

def get_resnet50(num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    for layer in base_model.layers:
        print("freezing layer {}".format(layer.name))
        layer.trainable = False

    x = base_model.output
    x = Dense(num_classes, activation='softmax')(x)
    model_classifier = Model(inputs=base_model.input, outputs=x)
    #print("model :")
    #model_classifier.summary()

    return model_classifier


def train_classifier(model, dataset, prep_func, optimzer, lr, net):
    train_data, train_y, val_data, val_y = dataset

    print("len of train = {}".format(len(train_data)))
    print("len of val = {}".format(len(val_data)))

    # define input data generators
    shift = 0.10
    datagen_train = ImageDataGenerator(rotation_range=15, width_shift_range=shift, height_shift_range=shift,
                                       preprocessing_function=prep_func,
                                       horizontal_flip=True, zoom_range=0.1)
    datagen_train.fit(train_data)

    # For validation, do not rotate. do less augmentation
    shift = 0.05
    datagen_test = ImageDataGenerator(preprocessing_function=prep_func,
                                      width_shift_range=shift, height_shift_range=shift,
                                      horizontal_flip=True, zoom_range=0.1)
    datagen_test.fit(val_data)

    epochs = 24
    batch_size = 32
    steps_per_epoch = int(len(train_data) / batch_size)
    steps_per_epoch_val = int(len(val_data) / batch_size)
    print("steps per epoch : {}, val : {}".format(steps_per_epoch, steps_per_epoch_val))

    if optimzer == 'adam':
        opt = Adam(lr=lr)
    elif optimzer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=True)
    elif optimzer == 'SGD':
        opt = 'SGD'


    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

#    model.summary()
    log_dir = os.path.join('logs', '_' + net + optimzer + '_' + str(lr) + ' _best_weights.h5')
    checkpoint = ModelCheckpoint(log_dir,
                                 monitor='val_loss', save_weights_only=True, save_best_only=True)

    model.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=0,
                        validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val)

    print("Adding reduceLR callback, early stop")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, cooldown=3)
#    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

#    model.compile(loss='categorical_crossentropy',
#                  optimizer=opt,
#                  metrics=['accuracy'])

    model.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=2 * epochs, initial_epoch=epochs,
                        validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val,
                        callbacks=[checkpoint, reduce_lr])
    print("restore best weights from checkpoint")
    model.load_weights(log_dir)

    return model


def train_post_classifier(lines, idxs_train, idxs_val):
    # prepare data
    lines = np.array(lines)
    train_data, y = process_lines(lines[idxs_train], net_type='resnet50')
    val_data, vy = process_lines(lines[idxs_val], net_type='resnet50')

    hist, bin_edges = np.histogram(y, bins=[0, 1, 2, 3, 4, 5, 6])
    print("hist train y:\n{}".format(hist))

    hist, bin_edges = np.histogram(vy, bins=[0, 1, 2, 3, 4, 5, 6])
    print("hist val y:\n{}".format(hist))

    num_classes = len(classes_list)
    cat_y = to_categorical(y, num_classes)
    cat_vy = to_categorical(vy, num_classes)
    print("num classes {}".format(num_classes))
    print("classes list {}".format(classes_list))
    # Grid search over optimizerss and lr's
    opts = ['adam', ]
    lrs = [0.01]

    for opt in opts:
        for lr in lrs:
            resnet50 = get_resnet50(num_classes)
            print("========= Training mobilenet1 with lr={}, opt={}========".format(lr, opt))
            train_classifier(resnet50, [train_data, cat_y, val_data, cat_vy], prep_func=preprocess_input, net='resnet50', optimzer=opt, lr=lr)
            print("===========================================")
            print("==== test resnet50 ====")
            print("y_gt : {}".format(vy))
            prep = preprocess_input(val_data)
            preds_resnet50 = resnet50.predict(prep)
            print("preds:\n{}".format(preds_resnet50))
            y_resnet50 = np.argmax(preds_resnet50, axis=1)
            print("\ty_resnet50:{}\n".format(y_resnet50))
            print("\nConfusion matrix:")
            print(confusion_matrix(vy, y_resnet50))
            print("\nClassification report:")
            print(classification_report(vy, y_resnet50))
            print("===========================================")


def predict_class(pil_image, boxes, classifier):
    if len(boxes) == 0:
        return []

    # Unpack classifiers (I know this is ineffective! should not be passed on stack but constant in bss ...)
    # if key does not exists, get returns None ...
    svm, vgg, mymobilenet = [classifier.get(name) for name in ['svm', 'vgg', 'mobilenet']]
    # VGG must be present!
    assert vgg is not None

    x_img_arr = []
    for box in boxes:
        # boxes is a list of [top, left, bottom, right] i.e [y1, x1, y2, x2]
        y1, x1, y2, x2 = box  # TODO
        cropped = pil_image.copy()  # TODO , Do I really need to copy ?
        # crop : left, upper, right, lower
        cropped = cropped.crop((x1, y1, x2, y2))
        cropped = cropped.resize((224,224))
        cropped = image.img_to_array(cropped)
        x_img_arr.append(cropped)

    x_img_arr = np.array(x_img_arr)

    vgg_preped = preprocess_input(x_img_arr)
    y_vgg, features = vgg.predict(vgg_preped)

    if svm is not None:
        y_svm = svm.predict(features)
    else:
        y_svm = np.argmax(y_vgg, axis=1)

    if mymobilenet is not None:
        x_mobilent = mobilenet.preprocess_input(x_img_arr)
        y_mobilenet = mymobilenet.predict(x_mobilent)
    else:
        y_mobilenet = y_vgg

    # Transform probabilities to labels (svm is alrady labels)
    y_mobilenet = np.argmax(y_mobilenet, axis=1)
    y_vgg = np.argmax(y_vgg, axis=1)

    # Do voting for final classification
    y = []
    for a, b, c in zip(y_mobilenet, y_vgg, y_svm):
        counts = np.bincount([a, b, c])
        y.append(np.argmax(counts))

    logging.debug("y mobilenet :{}".format(y_mobilenet))
    logging.debug("y_vgg       :{}".format(y_vgg))
    logging.debug("y_svm       :{}".format(y_svm))
    logging.debug("voting y    :{}".format(y))

    return y
# End


def main():
    print('unit testing')
    train_anns = '/home/tamirmal/workspace/git/tau_proj_prep/OUT_yolo_train_zero_based.txt'
    with open(train_anns) as f:
        train_lines = f.readlines()
        print("train len {}".format(len(train_lines)))

    test_anns = '/home/tamirmal/workspace/git/tau_proj_prep/OUT_yolo_test_zero_based.txt'
    with open(test_anns) as f:
        test_lines = f.readlines()
        print("test len {}".format(len(test_lines)))

    lines = list((*train_lines, *test_lines))
    print("lines shape {}".format(len(lines)))

    train_idx = len(train_lines)
    idxs_train = [i for i in range(train_idx)]
    idxs_val = [i for i in range(train_idx, len(lines))]

    train_post_classifier(lines, idxs_train, idxs_val)


if __name__ == "__main__":
    main()
