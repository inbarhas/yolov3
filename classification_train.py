import numpy as np
import PIL
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization, GlobalMaxPooling2D
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
            # need to make the image "square", because it will be reshaped later & we want to maintain the aspect ratio
            h = y2 - y1
            w = x2 - x1
            d = max(h, w)
            x1, y1, x2, y2 = x1, y1, x1 + d, y1 + d
            # crop : left, upper, right, lower
            copy_im = img.copy()
            cropped = copy_im.crop((x1, y1, x2, y2))
            cropped = cropped.resize(network_input_shape[net_type])
            sample = image.img_to_array(cropped)
            #sample = np.expand_dims(sample, axis=0)

            data.append(sample)
            y.append(gt)

            # Count how much classes we have
            if gt not in classes_list:
                classes_list.append(gt)

            ########## DEBUG HOOKS ############
            if visualize:
                from matplotlib import pyplot
                pyplot.figure()
                pyplot.imshow(cropped)
                pyplot.show()
            ###################################

    data = np.array(data)
    y = np.array(y)

    hist, bin_edges = np.histogram(y, bins=[0, 1, 2, 3, 4, 5, 6])
    print("hist y:\n{}".format(hist))

    return data, y
# End

########################################
#### VGG16
########################################
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
# End


def train_vgg_classifier(model, dataset):
    print("==== Training a VGG based classifier ====")
    train_data, train_y, val_data, val_y = dataset

    print("len of train = {}".format(len(train_data)))
    print("len of val = {}".format(len(val_data)))

    # define input data generators
    shift = 0.05
    datagen_train = ImageDataGenerator(rotation_range=10, width_shift_range=shift, height_shift_range=shift,
                                       preprocessing_function=preprocess_input,
                                       horizontal_flip=True, zoom_range=0.2)
    datagen_train.fit(train_data)

    # For validation, do not rotate. do less augmentation
    shift = 0.05
    datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input,
                                      width_shift_range=shift, height_shift_range=shift,
                                      horizontal_flip=True, zoom_range=0.2)
    datagen_test.fit(val_data)

    epochs = 200
    batch_size = 32
    steps_per_epoch = int(len(train_data) / batch_size)
    steps_per_epoch_val = int(len(val_data) / batch_size)
    print("steps per epoch : {}, val : {}".format(steps_per_epoch, steps_per_epoch_val))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    model.summary()
    log_dir = 'vgg_logs'
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'chkp_vgg_classifier_best.h5'),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    model.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=0,
                        validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val, callbacks=[checkpoint])

    print("Last steps - attemp to lower LR/early stop")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, cooldown=5)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    model.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=2 * epochs, initial_epoch=epochs,
                        validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val, callbacks=[checkpoint, early_stopping, reduce_lr])
    model.save_weights(os.path.join(log_dir, 'vgg_classifier_final.h5'))
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


"""
###############################################
#### Mobilenet
###############################################
def mobilenet1_get_model(num_classes):
    base_model = mobilenet.MobileNet(include_top=False, weights='imagenet',
                                     input_shape=(224, 224, 3))
    print("base")
    base_model.summary()

    x = Sequential()
    x.add(base_model)
    x.add(GlobalAveragePooling2D())
    x.add(Dropout(0.5))
    x.add(Dense(512))
    x.add(Dropout(0.5))
    x.add(Dense(num_classes, activation='sigmoid'))
    print("my additions")
    x.summary()
    return x
# End


def train_mobilenet1(model, dataset):
    train_data, y, val_data, vy = dataset

    # define input data generators
    shift = 0.1
    datagen_train = ImageDataGenerator(rotation_range=30, width_shift_range=shift, height_shift_range=shift,
                                       horizontal_flip=True, zoom_range=0.2)
    datagen_train.fit(train_data)

    # For validation, do not rotate. do less augmentation
    shift = 0.05
    datagen_test = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift,
                                      horizontal_flip=True, zoom_range=0.1)
    datagen_test.fit(val_data)

    epochs = 100
    batch_size = 32
    steps_per_epoch = int(len(train_data) / batch_size)
    steps_per_epoch_val = int(len(val_data) / batch_size)

    print("len data {}, len val {}".format(len(train_data), len(val_data)))
    print("steps per epoch : {}, val : {}".format(steps_per_epoch, steps_per_epoch_val))

    print("Freezing everything except last 5 layers")
    for layer in model.layers[:-5]:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit_generator(datagen_train.flow(train_data, y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=0,
                        validation_data=datagen_test.flow(val_data, vy, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val)

    print("unfreeze all model ...")
    log_dir = 'mobilenet_logs/'
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, cooldown=3)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)

    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit_generator(datagen_train.flow(train_data, y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=2 * epochs, initial_epoch=epochs,
                        validation_data=datagen_test.flow(val_data, vy, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val,
                        callbacks=[checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'mobilenet_trained_weights_final.h5')

# End


def prep_mobilenet(img_array):
    processed_image_mobilenet = mobilenet.preprocess_input(img_array.copy())
    return processed_image_mobilenet
"""
################################################
################################################


def train_post_classifier(lines, idxs_train, idxs_val):
    # prepare data
    lines = np.array(lines)
    train_data, y = process_lines(lines[idxs_train], net_type='vgg16')
    val_data, vy = process_lines(lines[idxs_val], net_type='vgg16')
    num_classes = len(classes_list)
    cat_y = to_categorical(y, num_classes)
    cat_vy = to_categorical(vy, num_classes)
    print("num classes {}".format(num_classes))
    print("classes list {}".format(classes_list))

    """
    ### Mobilenet ###
    my_mobilenet = mobilenet1_get_model(num_classes)
    train_mobilenet1(my_mobilenet, [train_data, cat_y, val_data, cat_vy])
    """

    ### VGG ###
    model_classifier, model_avg, model_max, vgg_all = vgg_get(num_classes)
    train_vgg_classifier(model_classifier, [train_data, cat_y, val_data, cat_vy])

    svm_max_f_data, svm_max_f_y, svm_avg_f_data, svm_avg_f_y = extract_vgg_features(vgg_all, [train_data, y])
    vgg_max_svm = train_svm([svm_max_f_data, svm_max_f_y], 'vgg', 'max')
    vgg_avg_svm = train_svm([svm_avg_f_data, svm_avg_f_y], 'vgg', 'avg')
    print("==== test VGG ====")
    prep_svm_val = preprocess_input(val_data)
    vgg_preds, features_max, features_avg = vgg_all.predict(prep_svm_val)
    print("\ty_vgg_classifier:\n".format(np.argmax(vgg_preds, axis=1)))
    print("testing vgg max svm")
    test_svm(vgg_max_svm, [features_max, vy])
    print("testing vgg avg svm")
    test_svm(vgg_avg_svm, [features_avg, vy])
    print("==== done testing vgg ====")


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

    """
    # raw outputs, some are softmax prob's
    if False:
        print("y mobilenet :{}".format(y_mobilenet))
        print("y_vgg       :{}".format(y_vgg))
        print("y_svm       :{}".format(y_svm))
    """

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
