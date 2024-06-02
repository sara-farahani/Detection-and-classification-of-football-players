from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# Prepare train and test data
def prepare_dataset():
    train_data_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2)
    test_data_generator = ImageDataGenerator(rescale=1. / 255)
    train = train_data_generator.flow_from_directory('train', target_size=(48, 36), batch_size=16, class_mode='binary', shuffle = True)
    test = test_data_generator.flow_from_directory('test', target_size=(48, 36), batch_size=16, class_mode='binary', shuffle = True)
    return train, test

    
# Define the model
def define_classification_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(48, 36, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def main():
    train_data, test_data = prepare_dataset()
    model = define_classification_model()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    # train the model
    model.fit(train_data, steps_per_epoch=128, epochs=40, validation_data=test_data, validation_steps=128)
    
    # save the model
    model.save("team_classifier_cnn.model")


if __name__ == "__main__":
    main()