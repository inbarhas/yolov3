import numpy as np
import PIL
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization
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
    return data, y
# End

########################################
#### VGG16
########################################

# Returns 2 models : One to be used as a feature extractor. One for classification
def vgg16_get_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    print("Base model VGG summary:")
    base_model.summary()

    print("Using base VGG as feature extractor")
    vgg_features = Model(inputs=base_model.input, outputs=base_model.output)

    x = Sequential()
    x.add(base_model)
    x.add(GlobalAveragePooling2D())
    x.add(Dense(128, activation='relu'))
    x.add(Dropout(0.5))
    x.add(Dense(num_classes, activation='softmax', name='predictions'))
    print("vgg - with my TOP layers")
    x.summary()

    return x, vgg_features
# End

def vgg_extract_features_and_prep(img_array, model):
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features, x

def vgg_extract_features_img_array(img_array, model):
    features, _ = vgg_extract_features_and_prep(img_array, model)
    return features
# End


# imgs array should already be a tensor
def vgg_extract_features_batch(imgs_array, model):
    prep = preprocess_input(imgs_array)
    features = model.predict(prep)
    return features, prep


def train_vgg(model, dataset):
    train_data, train_y, val_data, val_y = dataset

    print("len of train = {}".format(len(train_data)))
    print("len of val = {}".format(len(val_data)))

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
                  metrics=['accuracy', 'mae'])
    model.summary()
    model.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=0,
                        validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val)

    print("unfreeze all model ...")
    log_dir = 'vgg_logs/'
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, cooldown=3)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)

    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy', 'mae'])
    model.summary()
    model.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=2 * epochs, initial_epoch=epochs,
                        validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val,
                        callbacks=[checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'vgg_full_trained_weights_final.h5')

    return model


def predict_vgg(model, lines):
    train_data, y = process_lines(lines, 'vgg16')
    # process lines returns image as arrays
    data_to_predict = preprocess_input(train_data)
    y_predict = model.predict(data_to_predict)
    print("vgg full prediction")
    print("y=\n{}".format(y))
    print("y_predict=\n{}".format(y))


##############################################
### SVM over VGG features
##############################################

def train_svm(vgg_features, dataset):
    train_data, y, val_data, vy = dataset

    # Compile TODO not compiling for predictions
#    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
#    net.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print("==========================================")
    print("==== training SVM")
    print("==========================================")

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

    samples_train = 50 * len(train_data)
    print("Generating {} testing examples, using data aug".format(samples_train))
    svm_x_data = []
    svm_y_data = []
    cnt = 0
    for x_batch, y_batch in datagen_train.flow(train_data, y, batch_size=1):
        svm_x_data.append(vgg_extract_features_img_array(x_batch[0], vgg_features))
        svm_y_data.append(y_batch[0])
        cnt += 1
        if cnt > samples_train:
            break

    svm_x_data = np.array(svm_x_data)
    svm_y_data = np.array(svm_y_data)
    svm_x_data = np.reshape(svm_x_data, (len(svm_x_data), -1))

    # Param grid, after some tests I've seen that linear works well, and C=1.
    param = [
        {
            "kernel": ["linear"],
            "C": [1]
        },
    ]

    # request probability estimation
    svm = SVC(probability=True)
    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = GridSearchCV(svm, param, cv=4, n_jobs=4, verbose=3)
    # import ipdb; ipdb.set_trace()
    clf.fit(svm_x_data, svm_y_data)
    print("\nBest parameters set:")
    print(clf.best_params_)
    clf = clf.best_estimator_

    print("Run on test set :")
    samples_val = 20 * len(train_data)
    print("Generating {} testing examples, using data aug".format(samples_val))
    val_svm_x_data = []
    val_svm_y_data = []
    cnt = 0
    for x_batch, y_batch in datagen_train.flow(val_data, vy, batch_size=1):
        val_svm_x_data.append(vgg_extract_features_img_array(x_batch[0], vgg_features))
        val_svm_y_data.append(y_batch[0])
        cnt += 1
        if cnt > samples_val:
            break

    val_svm_x_data = np.array(val_svm_x_data)
    val_svm_y_data = np.array(val_svm_y_data)

    val_svm_x_data = np.reshape(val_svm_x_data, (len(val_svm_x_data), -1))
    y_predict = clf.predict(val_svm_x_data)

    print("y predicted")
    print(y_predict)
    print("val svm y data")
    print(val_svm_y_data)
    print("\nConfusion matrix:")
    print(confusion_matrix(val_svm_y_data, y_predict))
    print("\nClassification report:")
    print(classification_report(val_svm_y_data, y_predict))

    out_cls_path = 'svm.dump'
    print("Saving SVM to {}".format(out_cls_path))
    joblib.dump(clf, out_cls_path)
    print("done saving SVM")
    print("End of SVM training")
# End


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
                  metrics=['accuracy', 'mae'])
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
                  metrics=['accuracy', 'mae'])
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

################################################
################################################

def train_post_classifier(lines, idxs_train, idxs_val):
    # prepare data
    lines = np.array(lines)
    train_data, y = process_lines(lines[idxs_train], net_type='vgg16')
    val_data, vy = process_lines(lines[idxs_val], net_type='vgg16')
    num_classes = len(classes_list)
    print("num classes {}".format(num_classes))

    cat_y = to_categorical(y, num_classes)
    cat_vy = to_categorical(vy, num_classes)

    my_mobilenet = mobilenet1_get_model(num_classes)
    train_mobilenet1(my_mobilenet, [train_data, cat_y, val_data, cat_vy])

    vgg_classifier, vgg_features = vgg16_get_model(num_classes)
    train_vgg(vgg_classifier, [train_data, cat_y, val_data, cat_vy])
    ## TODO important, SVM used the trained VGG as feature-extractor ! not original with imagenet
    train_svm(vgg_features, [train_data, y, val_data, vy])


def predict_class(pil_image, boxes, classifier):
    if (len(boxes) == 0):
        return []

    svm = classifier['svm']
    vgg_features = classifier['vgg_features']
    my_mobilenet = classifier['mobilenet']
    vgg_classifier = classifier['vgg_classifier']

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
    y_vgg = vgg_classifier.predict(vgg_preped)
    features = vgg_features.predict(vgg_preped)
    features = np.reshape(features, (len(boxes), -1))

    x_mobilent = mobilenet.preprocess_input(x_img_arr)
    y_mobilenet = my_mobilenet.predict(x_mobilent)
    y_svm = svm.predict(features)

    # raw outputs, some are softmax prob's
    if True:
        print("y mobilenet :{}".format(y_mobilenet))
        print("y_vgg       :{}".format(y_vgg))
        print("y_svm       :{}".format(y_svm))

    y_mobilenet = np.argmax(y_mobilenet, axis=1)
    y_vgg = np.argmax(y_vgg, axis=1)

    # Do voting for final classification
    y = []
    for a, b, c in zip(y_mobilenet, y_vgg, y_svm):
        counts = np.bincount([a, b, c])
        y.append(np.argmax(counts))
    y = y_svm

    # debug hook
    if True:
        print("y mobilenet :{}".format(y_mobilenet))
        print("y_vgg       :{}".format(y_vgg))
        print("y_svm       :{}".format(y_svm))
        print("voting y    :{}".format(y))

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
