import numpy as np
import matplotlib as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

from data_loader import EMBEDDING_SIZE, MAX_SEQ_LEN_CONTENT, MAX_SEQ_LEN_SUMMARY, NUM_CLASSES
from data_loader import DataProcess
from model import Models

BATCH_SIZE = 32
EPOCHS = 10
NUM_FILTERS = 128
FILTER_SIZE = [3, 4, 5]
HIDDEN_SIZE = 32

if __name__ == '__main__':
    loader = DataProcess()
    train_x, test_x, train_y, test_y = loader.get_data()
    train_x_content = train_x[0]
    train_x_summary = train_x[1]
    test_x_content = test_x[0]
    test_x_summary = test_x[1]

    raw_label = loader.get_raw_label()

    num_words = loader.get_token_nums()
    embedding_matrix = loader.get_embedding_matrix()

    m = Models(NUM_CLASSES, num_words, MAX_SEQ_LEN_CONTENT, MAX_SEQ_LEN_SUMMARY, EMBEDDING_SIZE, embedding_matrix)
    model = m.multi_channel_model(NUM_FILTERS, FILTER_SIZE, HIDDEN_SIZE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=1e-5)
    history = model.fit(x=[train_x_content, train_x_summary],
                        y=train_y,
                        batch_size=BATCH_SIZE,
                        validation_data=([test_x_content, test_x_summary], test_y),
                        epochs=EPOCHS,
                        callbacks=[learning_rate_reduction],
                        verbose=1)

    print('Accuracy of the model on Training Data is - ',
          model.evaluate([train_x_content, train_x_summary], train_y)[1])
    print('Accuracy of the model on Testing Data is - ',
          model.evaluate([test_x_content, test_x_summary], test_y)[1])

    pred = model.predict((test_x_content, test_x_summary))
    pred = np.argmax(pred, axis=1)
    pred = [idx + 1 for idx in pred]

    print(type(pred))
    print(pred[:10])

    print(classification_report(raw_label, pred, target_names=['1', '2', '3', '4', '5']))

    cm = confusion_matrix(raw_label, pred)
    print(cm)

    epochs = [i for i in range(EPOCHS)]
    fig, ax = plt.subplots(1, 2)

    train_acc = history.history['acc']
    train_loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    fig.set_size_inches(20, 10)

    ax[0].plot(epochs, train_acc, label='Training Acc')
    ax[0].plot(epochs, val_acc, label='Testing Acc')
    ax[0].set_title('Training & Testing Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Acc")

    ax[1].plot(epochs, train_loss, label='Training Loss')
    ax[1].plot(epochs, val_loss, label='Testing Loss')
    ax[1].set_title('Training & Testing Loss')
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("Loss")

    plt.savefig('fig.png')

    # show badcase
    badcases = []
    for idx in range(len(pred)):
        if pred[idx] != raw_label[idx]:
            badcases.append((test_x_content[idx], test_x_summary[idx], raw_label[idx], pred[idx]))

    with open('badcase.tsv', 'r') as f:
        for badcase in badcases:
            f.write('\t'.joint(badcase) + '\n')
