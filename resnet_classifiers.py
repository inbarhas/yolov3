import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam
from classification_models import ResNet18
from classification_models import ResNet34
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import PIL
import os
import numpy as np


## Referenced from https://github.com/qubvel/classification_models, MIT license

classes_list = []

network_input_shape = {
    'resnet': (224, 224),
}


def preprocess_input(x, size=None, BGRTranspose=True):
    """input standardizing function
    Args:
        x: numpy.ndarray with shape (H, W, C)
        size: tuple (H_new, W_new), resized input shape
    Return:
        x: numpy.ndarray
    """

    if size:
        im = PIL.Image.fromarray(x)
        im = im.resize((size[1], size[0]))
        x = image.img_to_array(im, dtype=np.uint8)

    if BGRTranspose:
        x = x[..., ::-1]

    return x


def process_lines(lines, net_type='resnet'):
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
            sample = image.img_to_array(cropped, dtype=np.uint8)
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
            sample = image.img_to_array(cropped, dtype=np.uint8)
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


def get_resnet18(num_classes):
    # build model
    base_model = ResNet18(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = keras.layers.AveragePooling2D((7, 7))(base_model.output)
    x = keras.layers.Dropout(0.3)(x)
    output = keras.layers.Dense(num_classes)(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    #print("resnet18 summary:")
    #model.summary()
    return model
# End


def get_resnet34(num_classes):
    # build model
    base_model = ResNet34(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = keras.layers.AveragePooling2D((7, 7))(base_model.output)
    x = keras.layers.Dropout(0.3)(x)
    output = keras.layers.Dense(num_classes)(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    #print("resnet34 summary:")
    #model.summary()
    return model
# End


def train_classifier(model, dataset, net='resnetN'):
    print("==== Training a {} based classifier ====".format(net))
    train_data, train_y, val_data, val_y = dataset

    print("len of train = {}, shape {}".format(len(train_data), train_data.shape))
    print("len of val = {}, shape {}".format(len(val_data), val_data.shape))

    log_file = '_{}'.format(net)

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

    epochs = 64
    batch_size = 16
    steps_per_epoch = int(len(train_data) / batch_size)
    steps_per_epoch_val = int(len(val_data) / batch_size)
    print("steps per epoch : {}, val : {}".format(steps_per_epoch, steps_per_epoch_val))
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    log_dir = os.path.join('logs', log_file + '_best_weights.h5')
    checkpoint = ModelCheckpoint(os.path.join(log_dir),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    model.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=0,
                        validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val, callbacks=[checkpoint])

    print("Second step - add reduce LR callback & early stop")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, cooldown=5)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch, epochs=2 * epochs, initial_epoch=epochs,
                        validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                        validation_steps=steps_per_epoch_val, callbacks=[checkpoint, reduce_lr, early_stopping])
    print("Restoring best weights from checkpoint")
    model.load_weights(log_dir)
    print("==== end ====")
    return model
# End


def train_post_classifier(lines, idxs_train, idxs_val):
    # prepare data
    lines = np.array(lines)
    train_data, y = process_lines(lines[idxs_train], net_type='resnet')
    val_data, vy = process_lines(lines[idxs_val], net_type='resnet')

    hist, bin_edges = np.histogram(y, bins=[0, 1, 2, 3, 4, 5, 6])
    print("hist train y:\n{}".format(hist))

    hist, bin_edges = np.histogram(vy, bins=[0, 1, 2, 3, 4, 5, 6])
    print("hist val y:\n{}".format(hist))

    num_classes = len(classes_list)
    cat_y = to_categorical(y, num_classes)
    cat_vy = to_categorical(vy, num_classes)
    print("num classes {}".format(num_classes))
    print("classes list {}".format(classes_list))

    resnet18 = get_resnet18(num_classes)
    resnet34 = get_resnet34(num_classes)
    print("========= Training resnet18")
    train_classifier(resnet18, [train_data, cat_y, val_data, cat_vy], 'resnet18')
    print("========= Training resnet34")
    train_classifier(resnet34, [train_data, cat_y, val_data, cat_vy], 'resnet34')
    print("===========================================")
    prep = preprocess_input(val_data.copy)
    print("==== test resnet18 ====")
    print("y_gt : {}".format(vy))
    preds_18 = resnet18.predict(prep)
    print("preds_18:\n{}".format(preds_18))
    preds = np.argmax(preds_18, axis=1)
    print("\ty_18:{}\n".format(preds))
    print("\nConfusion matrix:")
    print(confusion_matrix(vy, preds))
    print("\nClassification report:")
    print(classification_report(vy, preds))
    print("===========================================")
    print("==== test resnet34 ====")
    print("y_gt : {}".format(vy))
    preds_34 = resnet34.predict(prep)
    print("preds_34:\n{}".format(preds_34))
    preds = np.argmax(preds_34, axis=1)
    print("\ty_34:{}\n".format(preds))
    print("\nConfusion matrix:")
    print(confusion_matrix(vy, preds))
    print("\nClassification report:")
    print(classification_report(vy, preds))
    print("===========================================")
    print("==== avg 18 & 34 ====")
    print("y_gt : {}".format(vy))
    preds_avg = (preds_18 + preds_34) / 2
    print("avg preds:\n{}".format(preds_avg))
    preds = np.argmax(preds_avg, axis=1)
    print("\ty_avg:{}\n".format(preds))
    print("\nConfusion matrix:")
    print(confusion_matrix(vy, preds))
    print("\nClassification report:")
    print(classification_report(vy, preds))

    print("--- END ---")


from skimage.io import imread
def main():

    x = imread('./Dog.jpg')
    xx = PIL.Image.open('./Dog.jpg')
#    cropped = xx.resize(network_input_shape['resnet'])
    sample = image.img_to_array(xx, dtype=np.uint8)

    aaa = preprocess_input(sample, (224,224), True)

    from matplotlib import pyplot
    pyplot.figure()
    pyplot.imshow(xx)
    pyplot.show()

    from matplotlib import pyplot
    pyplot.figure()
    pyplot.imshow(aaa)
    pyplot.show()

    sample_preped = np.expand_dims(sample, 0)
    sample_preped = keras.applications.resnet50.preprocess_input(sample_preped)

    print("end")

    # read and prepare image
#    x = imread('./imgs/tests/seagull.jpg')
#    x = resize(x, (224, 224)) * 255  # cast back to 0-255 range
#    x = preprocess_input(x)
#    x = np.expand_dims(x, 0)

if __name__ == "__main__":
    main()