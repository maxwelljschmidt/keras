import tensorflow as tf
from tensorflow import keras
from collections import Counter


def getdataset(training=True):
    mnist = keras.datasets.mnist #create dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    if training:  # return depending on training Var
        return train_images, train_labels
    return test_images, test_labels


def print_stats(train_images, train_labels):
    print(len(train_images))  # print length
    print(len(train_images[0]), 'x', len(train_images[0][0]), sep='')  # print image dimensions

    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    count = Counter(train_labels)  # use counter class to efficently create list of occurances
    count = sorted(count.items())  # sort occurances list
    for a in range(len(count)):  # print occurances as specified in writeup
        print(a, ". ", class_names[a], " - ", count[a][1], sep='')


def build_model():
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28))  # flatten layer
        ]
    )  # initiate model

    optimize = keras.optimizers.SGD(learning_rate=0.001)  # initiate model parameters per specification
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # loss function
    metric = ['accuracy']  # piazza @1616, metric

    model.add(keras.layers.Dense(128, activation=keras.activations.relu))  # 128 node dense layer

    model.add(keras.layers.Dense(64, activation=keras.activations.relu))  # 64 node dense layer

    model.add(keras.layers.Dense(10))  # 10 node dense layer

    model.compile(optimize, loss, metric)  # compile with model parameters
    return model  # return untrained model


def train_model(model, train_images, train_labels, T):
    model.fit(train_images, train_labels, epochs=T)


def evaluate_model(model, test_images, test_labels, show_loss=True):
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

    txt = "Accuracy: {:.2%}"  # accuracy return string, float formatting
    if show_loss:
        print("Loss:", '%.4f' % test_loss)  # if true, return loss with formatting
    print(txt.format(test_accuracy)) #print with formatting from above


def predict_label(model, test_images, index):
    m = model.predict(test_images) #use predict to create array of prediction values
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

    percentiles = list(zip(class_names, m[index])) #create zip list of class_names and the prediction values for index
    percentiles.sort(key=lambda x: x[1], reverse=True) #sort zip list by percentile values, descending

    for i in range(3):
        print(percentiles[i][0],": ", "{0:.2f}%".format(percentiles[i][1]*100), sep="") #print top 3 likely class labels


def main():
    test_images, test_labels = getdataset(False)
    train_images, train_labels = getdataset()
    print_stats(train_images, train_labels)

    m = build_model()
    train_model(m, train_images, train_labels, 10)

    evaluate_model(m, test_images, test_labels, True)
    m.add(keras.layers.Softmax())
    predict_label(m, train_images, 10)


if __name__ == '__main__':
    main()
