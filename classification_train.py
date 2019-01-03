import numpy as np
import PIL
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization, GlobalMaxPooling2D, Reshape, Conv2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from keras.applications import mobilenet
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import logging
import os
from keras import backend as K
from keras.applications import mobilenet_v2


classes_list = []

network_input_shape = {
    'vgg16': (224, 224),
    'mobilenet': (224, 224)
}


def process_lines(lines, net_type='vgg16'):
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


def vgg_get(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    classifier = base_model.output
    classifier = GlobalAveragePooling2D()(classifier)
    classifier = Dense(128, activation='relu')(classifier)
    classifier = Dropout(0.5)(classifier)
    classifier = Dense(num_classes, activation='softmax', name='predictions')(classifier)
    model_classifier = Model(inputs=base_model.input, outputs=classifier)
    print("model classifier summary:\n")
    model_classifier.summary()

    return model_classifier

    """
    max_features = base_model.output
    max_features = GlobalMaxPooling2D(name='svm_feed_max')(max_features)
    model_max = Model(inputs=base_model.input, outputs=max_features)
    print("model max features:\n")
    model_max.summary()

    avg_features = base_model.output
    avg_features = GlobalAveragePooling2D(name='svm_feed_avg')(avg_features)
    model_avg = Model(inputs=base_model.input, outputs=avg_features)
    print("model avg features:\n")
    model_avg.summary()

    three_heads = Model(inputs=base_model.input, outputs=[model_classifier.output, model_max.output, model_avg.output])
    print("3 head model summary:\n")
    three_heads.summary()
    
    return (model_classifier, model_avg, model_max, three_heads)
    """
# End


def mobilenet1_get(num_classes):
    base_model = mobilenet.MobileNet(include_top=False, weights='imagenet',
                                     input_shape=(224, 224, 3))
    # freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    alpha = 1.0
    dropout = 0.5

    if K.image_data_format() == 'channels_first':
        shape = (int(1024 * alpha), 1, 1)
    else:
        shape = (1, 1, int(1024 * alpha))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Reshape(shape, name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(num_classes, (1, 1),
               padding='same',
               name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((num_classes,), name='reshape_2')(x)

    model_classifier = Model(inputs=base_model.input, outputs=x)
    print("model classifier summary:\n")
    model_classifier.summary()
    return model_classifier
    """
    max_features = base_model.output
    max_features = GlobalMaxPooling2D(name='svm_feed_max')(max_features)
    model_max = Model(inputs=base_model.input, outputs=max_features)

    avg_features = base_model.output
    avg_features = GlobalAveragePooling2D(name='svm_feed_avg')(avg_features)
    model_avg = Model(inputs=base_model.input, outputs=avg_features)

    three_heads = Model(inputs=base_model.input, outputs=[model_classifier.output, model_max.output, model_avg.output])
    # print("3 head model summary:\n")
    # three_heads.summary()

    return (model_classifier, model_avg, model_max, three_heads)
    """
# End


def mobilenet2_get(num_classes):
    base_model = mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet',
                                          input_shape=(224, 224, 3))
    # freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5, name='dropout')(x)
    x = Dense(num_classes, activation='softmax',
              use_bias=True, name='Logits')(x)
    model_classifier = Model(inputs=base_model.input, outputs=x)
    model_classifier.summary()
    return model_classifier
    """
    max_features = base_model.output
    max_features = GlobalMaxPooling2D(name='svm_feed_max')(max_features)
    model_max = Model(inputs=base_model.input, outputs=max_features)

    avg_features = base_model.output
    avg_features = GlobalAveragePooling2D(name='svm_feed_avg')(avg_features)
    model_avg = Model(inputs=base_model.input, outputs=avg_features)

    three_heads = Model(inputs=base_model.input, outputs=[model_classifier.output, model_max.output, model_avg.output])
    # print("3 head model summary:\n")
    # three_heads.summary()

    return (model_classifier, model_avg, model_max, three_heads)
    """
# End


def train_classifier(model, dataset, prep_f, net='vgg', opt='adam', lr=0.001, unfreeze=False):
    print("==== Training a {} based classifier, opt={}, lr={} ====".format(net,opt, lr))
    train_data, train_y, val_data, val_y = dataset

    print("len of train = {}".format(len(train_data)))
    print("len of val = {}".format(len(val_data)))

    # define input data generators
    shift = 0.05
    datagen_train = ImageDataGenerator(rotation_range=10, width_shift_range=shift, height_shift_range=shift,
                                       preprocessing_function=prep_f,
                                       horizontal_flip=True, zoom_range=0.2)
    datagen_train.fit(train_data)

    # For validation, do not rotate. do less augmentation
    shift = 0.05
    datagen_test = ImageDataGenerator(preprocessing_function=prep_f,
                                      width_shift_range=shift, height_shift_range=shift,
                                      horizontal_flip=True, zoom_range=0.2)
    datagen_test.fit(val_data)

    epochs = 64
    batch_size = 16
    steps_per_epoch = int(len(train_data) / batch_size)
    steps_per_epoch_val = int(len(val_data) / batch_size)
    print("steps per epoch : {}, val : {}".format(steps_per_epoch, steps_per_epoch_val))

    if opt == 'adam':
        optz = Adam(lr=lr)
    elif opt == 'sgd':
        optz = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optz,
                  metrics=['accuracy'])

    model.summary()
    log_dir = os.path.join('logs', 'classifier_' + net + '_' + opt + '_' + str(lr) + '_best_weights.h5')
    checkpoint = ModelCheckpoint(os.path.join(log_dir),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    model.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=0,
                        validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val, callbacks=[checkpoint])

    print("Second step - add reduce LR callback. unfreeze last conv block")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, cooldown=5)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    if unfreeze:
        ######## Layers unfreezing ########
        def getLayerIndexByName(model, layername):
            for idx, layer in enumerate(model.layers):
                if layer.name == layername:
                    return idx

        if net == 'vgg':
            unfreeze_idx = getLayerIndexByName(model, 'block5_conv1')
        elif net == 'mobilenet1':
            unfreeze_idx = getLayerIndexByName(model, 'conv_dw_13')
        elif net == 'mobilenet2':
            unfreeze_idx = getLayerIndexByName(model, 'block_16_expand')

        for layer in model.layers[unfreeze_idx:]:
            print("unfreeze layer {}".format(layer.name))
            layer.trainable = True
        #####################################
    model.compile(loss='categorical_crossentropy',
                  optimizer=optz,
                  metrics=['accuracy'])

    model.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=2 * epochs, initial_epoch=epochs,
                        validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val, callbacks=[checkpoint, reduce_lr])
    print("Restoring best weights from checkpoint")
    model.load_weights(log_dir)
    print("==== end ====")
    return model
# End


def train_svm(dataset, net_type, pool_type):
    train_data, train_y = dataset
    print("==========================================")
    print("==== training SVM : {}, {}".format(net_type, pool_type))
    print("==========================================")
    print("len of train x {}, y {}".format(len(train_data), len(train_y)))

    param = [
        {
            "kernel": ["linear", "poly"],
            "C": [1, 10, 100, 1000],
            "gamma": ['scale', 'auto'],
        },
    ]

    # request probability estimation
    svm = SVC()
    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = GridSearchCV(svm, param, cv=10, n_jobs=4, verbose=3)
    clf.fit(train_data, train_y)
    print("\nBest parameters set:")
    print(clf.best_params_)
    clf = clf.best_estimator_

    out_cls_path = 'svm_' + net_type + '_' + pool_type
    print("Saving SVM classsifier at : {}".format(out_cls_path))
    joblib.dump(clf, out_cls_path)

    return clf
# End


def test_svm(svm, val_dataset):
    val_data, val_y = val_dataset
    y_predict = svm.predict(val_data)
    print("y predicted")
    print(y_predict)
    print("gt y")
    print(val_y)
    print("\nConfusion matrix:")
    print(confusion_matrix(val_y, y_predict))
    print("\nClassification report:")
    print(classification_report(val_y, y_predict))

    return y_predict


def extract_vgg_features(model, dataset):
    print("==== Extracting AVG & MAX VGG Features ====")
    train_data, train_y = dataset

    # define input data generators
    shift = 0.05
    datagen_train = ImageDataGenerator(rotation_range=10, width_shift_range=shift, height_shift_range=shift,
                                       preprocessing_function=preprocess_input,
                                       horizontal_flip=True, zoom_range=0.2)
    datagen_train.fit(train_data)
    samples_train = 20 * len(train_data)
    print("will generate total of {} samples with data-aug".format(samples_train))

    svm_max_f_data = []
    svm_avg_f_data = []
    svm_y_data = []

    cnt = 0
    for x_batch, y_batch in datagen_train.flow(train_data, train_y, batch_size=1):
        _, max_f, avg_f = model.predict(x_batch)
        svm_max_f_data.append(max_f)
        svm_avg_f_data.append(avg_f)

        svm_y_data.append(y_batch[0])

        cnt += 1
        if cnt > samples_train:
            break

    print("len of x avg : {}".format(len(svm_avg_f_data)))
    print("len of x max : {}".format(len(svm_max_f_data)))
    print("len of y : {}".format(len(svm_y_data)))

    svm_y_data = np.array(svm_y_data)

    svm_max_f_data = np.array(svm_max_f_data)
    svm_max_f_data = np.reshape(svm_max_f_data, (len(svm_max_f_data), -1))

    svm_avg_f_data = np.array(svm_avg_f_data)
    svm_avg_f_data = np.reshape(svm_avg_f_data, (len(svm_avg_f_data), -1))

    print("shape of x max : {}".format(svm_max_f_data.shape))
    print("shape of x avg : {}".format(svm_avg_f_data.shape))
    print("shape of y : {}".format(svm_y_data.shape))

    print("shape of x avg[0] : {}".format(svm_avg_f_data[0].shape))
    print("shape of x max[0] : {}".format(svm_max_f_data[0].shape))
    print("shape of y[0] : {}".format(svm_y_data[0].shape))

    return svm_max_f_data, svm_y_data, svm_avg_f_data, svm_y_data
# END


def train_post_classifier(lines, idxs_train, idxs_val):
    # prepare data
    lines = np.array(lines)
    train_data, y = process_lines(lines[idxs_train], net_type='vgg16')
    val_data, vy = process_lines(lines[idxs_val], net_type='vgg16')

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
    opts = ['adam', 'sgd']
    lrs = [0.01, 0.001]
    unfz = [True, False]

    for u in unfz:
        for opt in opts:
            for lr in lrs:
 #               vgg_classifier = vgg_get(num_classes)
                mobilenet1_classifier = mobilenet1_get(num_classes)
                mobilenet2_classifier = mobilenet1_get(num_classes)
 #               print("========= Training VGG with lr={}, opt={} unfreeze={}=========".format(lr, opt, u))
 #               train_classifier(vgg_classifier, [train_data, cat_y, val_data, cat_vy], preprocess_input, 'vgg', opt, lr)
                print("========= Training mobilenet1 with lr={}, opt={} unfreeze={}=========".format(lr, opt, u))
                train_classifier(mobilenet1_classifier, [train_data, cat_y, val_data, cat_vy], mobilenet.preprocess_input, 'mobilenet1', opt, lr)
                print("========= Training mobilenet2 with lr={}, opt={} unfreeze={}=========".format(lr, opt, u))
                train_classifier(mobilenet2_classifier, [train_data, cat_y, val_data, cat_vy], mobilenet_v2.preprocess_input, 'mobilenet2', opt, lr)
                print("==========================================================")
  #              print("==== test VGG ====")
  #              print("y_gt : {}".format(vy))
  #              prep = preprocess_input(val_data.copy())
  #              preds_vgg = vgg_classifier.predict(prep)
  #              print("preds vgg:\n{}".format(preds_vgg))
  #              preds = np.argmax(preds_vgg, axis=1)
  #              print("\ty_vgg:{}\n".format(preds))
  #              print("\nConfusion matrix:")
  #              print(confusion_matrix(vy, preds))
  #              print("\nClassification report:")
  #              print(classification_report(vy, preds))
                print("===========================================")
                print("==== test mobilenet1 ====")
                print("y_gt : {}".format(vy))
                prep = mobilenet.preprocess_input(val_data.copy())
                preds_m1 = mobilenet1_classifier.predict(prep)
                print("preds m1:\n{}".format(preds_m1))
                preds = np.argmax(preds_m1, axis=1)
                print("\ty_mobilenet1:{}\n".format(preds))
                print("\nConfusion matrix:")
                print(confusion_matrix(vy, preds))
                print("\nClassification report:")
                print(classification_report(vy, preds))
                print("===========================================")
                print("==== test mobilenet2 ====")
                print("y_gt : {}".format(vy))
                prep = mobilenet_v2.preprocess_input(val_data.copy())
                preds_m2 = mobilenet2_classifier.predict(prep)
                print("preds m2:\n{}".format(preds_m2))
                preds = np.argmax(preds_m2, axis=1)
                print("\ty_mobilenet2:{}\n".format(preds))
                print("\nConfusion matrix:")
                print(confusion_matrix(vy, preds))
                print("\nClassification report:")
                print(classification_report(vy, preds))
                print("===========================================")
                print("average between m1 & m2 ===================")
                preds = (preds_m1 + preds_m2)/2
                preds = np.argmax(preds, axis=1)
                print("\tavg_m1_m2:{}\n".format(preds))
                print("\nConfusion matrix:")
                print(confusion_matrix(vy, preds))
                print("\nClassification report:")
                print(classification_report(vy, preds))
                print("===========================================")
                print("===========================================")

    """
    svm_max_f_data, svm_max_f_y, svm_avg_f_data, svm_avg_f_y = extract_vgg_features(vgg_all, [train_data, y])
    vgg_max_svm = train_svm([svm_max_f_data, svm_max_f_y], 'vgg', 'max')
    vgg_avg_svm = train_svm([svm_avg_f_data, svm_avg_f_y], 'vgg', 'avg')
    """

    """
    print("testing vgg max svm")
    y_vgg_max = test_svm(vgg_max_svm, [features_max, vy])
    print("\nConfusion matrix:")
    print(confusion_matrix(vy, y_vgg_max))
    print("\nClassification report:")
    print(classification_report(vy, y_vgg_max))
    print("testing vgg avg svm")
    y_vgg_avg = test_svm(vgg_avg_svm, [features_avg, vy])
    print("\nConfusion matrix:")
    print(confusion_matrix(vy, y_vgg_avg))
    print("\nClassification report:")
    print(classification_report(vy, y_vgg_avg))
    print("============================================")
    print("final after voting")
    # Do voting for final classification
    y_voting = []
    for a, b, c in zip(y_vgg_max, y_vgg_avg, vgg_preds):
        counts = np.bincount([a, b, c])
        y_voting.append(np.argmax(counts))
    print("y_vote:{}".format(y_voting))
    print("y_gt  :{}".format(vy))
    print("\nConfusion matrix:")
    print(confusion_matrix(vy, y_voting))
    print("\nClassification report:")
    print(classification_report(vy, y_voting))
    print("==== done testing vgg ====")
    """

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
