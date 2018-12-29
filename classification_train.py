from keras.preprocessing import image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import PIL
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os.path


classes_list = []

network_input_shape = {
    'vgg16': (224, 224)
}


def process_lines(lines, type):
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
            cropped = cropped.resize(network_input_shape[type])
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


from keras.applications.inception_v3 import InceptionV3
def inception_v3_get_model(num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    img_width, img_height = 299, 299  # Inception image size
    top_layers_checkpoint_path = 'cp.top.best.hdf5'
    fine_tuned_checkpoint_path = 'cp.fine_tuned.best.hdf5'
    new_extended_inception_weights = 'final_weights.hdf5'

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- we have 2 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    assert 0  # TODO


from keras.applications import MobileNet
def mobilenet_get_model(num_classes):
    base_model = MobileNet(weights='imagenet', include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    preds = Dense(num_classes, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)

    # freeze all layers
    for layer in model.layers:
        layer.trainable = False
    # or if we want to set the first 20 layers of the network to be non-trainable, rest are trainable
    for layer in model.layers[20:]:
        layer.trainable = True

    print("Print model layers")
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    # compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# End


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
def vgg16_get_model(num_classes):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    # freeze all layers
    for layer in vgg.layers:
        layer.trainable = False

    vgg.summary()

    x = vgg.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    my_model = Model(inputs=vgg.input, outputs=x)
    my_model.summary()

    return my_model
# End


def train_post_classifier(lines, idxs_train, idxs_val, type='vgg16'):
    # prepare data
    train_data, train_y = process_lines(lines[idxs_train], type)
    val_data, val_y = process_lines(lines[idxs_val], type)

    # prepare model
    num_train = len(idxs_train)
    num_val = len(idxs_val)
    num_classes = len(classes_list)
    top_epochs = 50

    print("num classes {}".format(num_classes))
    train_y = to_categorical(train_y, num_classes)
    val_y = to_categorical(val_y, num_classes)


    # Get model
    if type == 'vgg16':
        net = vgg16_get_model(num_classes)
    else:
        print("only vgg16 for now")
        assert 0

    # Compile
    if type == 'vgg16':
        sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
        net.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("only vgg16 for now")
        assert 0

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

    print("==== Starting training - all layers but last are freezed ====")
    batch_size = 32
    epochs = 50
    steps_per_epoch = num_train/batch_size
    steps_per_epoch_val = num_val/batch_size
    net.fit_generator(datagen_train.flow(train_data, train_y, batch_size=batch_size),
                      steps_per_epoch=steps_per_epoch, epochs=epochs,
                      validation_data=datagen_test.flow(val_data, val_y, batch_size=batch_size),
                      validation_steps=steps_per_epoch_val)
    net.save_weights('post_vgg16.h5')

# End


def main():
    print('unit testing')
    annotation_path = 'unit_test.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
        lines = np.array(lines)

    val_split = 0.1
    val_idx = int(val_split * len(lines))

    idxs_train = [i for i in range(val_idx)]
    idxs_val = [i for i in range(val_idx, len(lines))]
    train_post_classifier(lines, idxs_train, idxs_val, type='vgg16')


if __name__ == "__main__":
    main()
