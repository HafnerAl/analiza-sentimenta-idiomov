import tensorflow as tf
import tensorflow as tf
import numpy as np
import sys
import torch
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, TimeDistributed, Masking, GRU
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from ast import literal_eval
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from random import shuffle
from transformers import BertTokenizer, BertModel
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.data.tokenizers import WhitespaceTokenizer
from tensorflow.keras.layers import Input

options_file = './models/options.json' #[r'E:\\FERI\\Magisterij\\JEZ TEH\SV\\analiza-sentimenta-idiomov\\models\\options.json']
weight_file = './models/slovenian-elmo-weights.hdf5' #[r'E:\\FERI\\Magisterij\\JEZ TEH\SV\\analiza-sentimenta-idiomov\\models\\slovenian-elmo-weights.hdf5']

elmo = Elmo(options_file, weight_file, num_output_representations=1)
tokenizer = WhitespaceTokenizer()


#VECTOR_DIM = 1024
VECTOR_DIM = 768 # bert
MAX_SEQUENCE_LEN = 50

IN_FILENAMES_TRAIN = './models/classes_elmo_1_5.txt' #[r'E:\\FERI\\Magisterij\\JEZ TEH\SV\\analiza-sentimenta-idiomov\\test_datasets\\classes_elmo_1_1.txt']
                      #r'E:\\FERI\\Magisterij\\JEZ TEH\SV\\analiza-sentimenta-idiomov\\test_datasets\\classes_elmo_2.txt']
IN_FILENAMES_TEST = './models/classes_elmo_1_5.txt' #[r'E:\\FERI\\Magisterij\\JEZ TEH\SV\\analiza-sentimenta-idiomov\\test_datasets\\classes_elmo_1_1.txt']
IN_FILENAME_TEST = None
NUM_CLASSES = 4


def get_xy_per_expression(filename):
    data_by_expressions = {}
    sent_wide_Y = []
    sents_X = []
    sents_Y = []
    curr_sent_X = []
    curr_sent_Y = []
    expressions = []
    print('starting')
    CLS_TO_INT_DICT = {'NE': 3, 'DA': 2, '*':1, 'NEJASEN_ZGLED':4}
    classes = []
    words = []
    X = []
    Y = []

    print('reading file', filename)
    with open(filename, 'r', encoding='utf-8') as f:
        debug_sent = []
        for i, line in enumerate(f):
            if i % 500 == 0:
                print(i)
            parts = line.split('\t')
            word = parts[0]
            if len(word) == 0:
                continue
            if len(parts) != 3:
                continue
            exp = parts[0]
            sentence = tokenizer.tokenize(exp)
            sentence_text = [token.text for token in sentence]
            
            character_ids = batch_to_ids([sentence_text])

            embeddings = elmo(character_ids)
            elmo_embedding = embeddings['elmo_representations'][0].detach().numpy()
            vector = elmo_embedding
            cls = parts[1]
            expression = parts[2]
            debug_sent.append((word, cls, expression))
            classes.append(cls)
            words.append(word)
            if not (cls == 'DA' or cls == 'NE' or cls == '*'):
                continue
            curr_sent_X.append(vector)
            curr_sent_Y.append(CLS_TO_INT_DICT[cls]) 
            if word[-1] == '.':
                if expression not in data_by_expressions.keys():
                    data_by_expressions[expression] = [(np.array(curr_sent_X), np.array(curr_sent_Y))]
                else:
                    data_by_expressions[expression].append((np.array(curr_sent_X), np.array(curr_sent_Y)))
                sent_wide_cls = None
                if CLS_TO_INT_DICT['DA'] in curr_sent_Y:
                    sent_wide_cls = CLS_TO_INT_DICT['DA']
                elif CLS_TO_INT_DICT['NE'] in curr_sent_Y:
                    sent_wide_cls = CLS_TO_INT_DICT['NE']
                else:
                    sent_wide_cls = CLS_TO_INT_DICT['NEJASEN_ZGLED']
                    print('debug sent', debug_sent)
                sent_wide_Y.append(sent_wide_cls)
                debug_sent = []
                curr_sent_X = []
                curr_sent_Y = []
        X = np.array(X)
        Y = np.array(Y)
        sents_X = np.array(sents_X)
        sents_Y = np.array(sents_Y)
        
    print(Counter(classes))
    print(Counter(words))
    print(X.shape)
    print(Y.shape)
    print(sents_X.shape)
    print(sents_Y.shape)
    #return sents_X, sents_Y
    return data_by_expressions

def bert_tensorflow_test(X_train, X_test, Y_train, Y_test):
    # Model
   # model = Sequential()
   # model.add(Masking(mask_value=0., input_shape=(MAX_SEQUENCE_LEN,VECTOR_DIM)))
    #forward_layer = LSTM(200, return_sequences=True)
    forward_layer = GRU(10, return_sequences=False, dropout=0.5)
    #backward_layer = LSTM(200, activation='relu', return_sequences=True,
    backward_layer = GRU(10, return_sequences=False, dropout=0.5,
                       go_backwards=True)
   # model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
   #                      input_shape=(MAX_SEQUENCE_LEN,VECTOR_DIM)))
    #model.add(TimeDistributed(Dense(NUM_CLASSES)))
    # Remove TimeDistributed() so that predictions are now made for the entire sentence
   # model.add(Dense(NUM_CLASSES))
   # model.add(Activation('softmax'))

    
    model = Sequential()
    model.add(Input(shape=(50, 1024)))  
    model.add(Bidirectional(GRU(10, return_sequences=False, dropout=0.5)))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #print('preds shape', model.predict(X_train[:3]).shape)
    #print('Y_train shape', Y_train[:3].shape)
    #print(list(Y_train[:3]))
    classes = []
    for y in Y_train:
        cls = np.argmax(y)
        classes.append(cls)
    print(Counter(classes))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('compiled model')
    model.fit(X_train, Y_train, batch_size=8, epochs=10)#, validation_split=0.1)
    print('fit model')
    eval = model.evaluate(X_test, Y_test, batch_size=8)
    preds = model.predict(X_test, verbose=1, batch_size=8)
    print(preds)
    num_correct = 0
    num_incorrect = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # idiomatic = 2, non-idiomatic = 3
    with open('preds_out_temp.txt', 'w') as tempoutf:
        for pred, y in zip(preds, Y_test):
            if np.argmax(y) == 2 or np.argmax(y) == 3:
                if np.argmax(y) == np.argmax(pred):
                    num_correct += 1
                else:
                    num_incorrect += 1
            if np.argmax(pred) == 2 and np.argmax(y) == 2:
                TP += 1
            if np.argmax(pred) == 3 and np.argmax(y) == 3:
                TN += 1
            if np.argmax(pred) == 2 and np.argmax(y) == 3:
                FP += 1
            if np.argmax(pred) == 3 and np.argmax(y) == 2:
                FN += 1
    custom_accuracy = num_correct/(num_correct+num_incorrect)
    print('custom accuracy is', num_correct/(num_correct+num_incorrect))
    for y in Y_test:
        cls = np.argmax(y)
        classes.append(cls)
    class_nums = Counter(classes)
    print(class_nums)
    default_acc = class_nums[2] / (class_nums[2] + class_nums[3])
    print('default accuracy is', default_acc, 'or', 1 - default_acc)
    f1s = []

    if TP == 0:
        precision = 0
        recall = 0
        print('precision', 0, file=outf)
        print('recall', 0, file=outf)
        print('F1 score', 0)
        f1s.append(0)
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        print('precision', TP/(TP+FP), file=outf)
        print('recall', TP/(TP+FN), file=outf)
        print('F1 score', (2*precision*recall)/(precision+recall))
        f1s.append((2*precision*recall)/(precision+recall))
        
    print('F1 average', sum(f1s)/len(f1s))
    return eval, custom_accuracy, default_acc, [TP, TN, FP, FN]
    

def get_already_processed(filename):
    if filename == None:
        return set([])
    already_processed = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.split(' ')
            if words[0] == 'EXP':
                already_processed.append(' '.join(words[1:]))
    return set(already_processed)

dbe = get_xy_per_expression(IN_FILENAMES_TRAIN)
print(dbe.keys())

with open('./test_results_elmo_shuffled.txt', 'w', encoding='utf-8') as outf:
    for k, i in dbe.items():
        print(k, len(i))
    
    for k, i in dbe.items():
        all_data = [item for sublist in dbe.values() for item in sublist]

        all_X = [x[0] for x in all_data]
        all_Y = [x[1] for x in all_data]

        max_len = max(len(x) for x in all_X)
        max_len = 50
        padded_all_X = pad_sequences(all_X, maxlen=max_len, padding='post', dtype='float32')

        all_X = [x[0].reshape(-1, 1024) for x in all_data]  
        max_len = max(x.shape[0] for x in all_X) 
        padded_all_X = pad_sequences(all_X, maxlen=max_len, dtype='float32') 


        train_X, test_X, train_Y, test_Y = train_test_split(padded_all_X, all_Y, test_size=0.30, shuffle=True)

        sent_train_Y = []
        sent_test_Y = []

        for y in train_Y:
            #print(y)
            if 2 in y:
                sent_train_Y.append(2)
            elif 3 in y:
                sent_train_Y.append(3)
            elif 4 in y:
                sent_train_Y.append(4)
            else:
                sent_train_Y.append(1)

        for y in test_Y:
            #print(y)
            if 2 in y:
                sent_test_Y.append(2)
            elif 3 in y:
                sent_test_Y.append(3)
            elif 4 in y:
                sent_test_Y.append(4)
            else:
                sent_test_Y.append(1)

        
        train_Y = to_categorical(sent_train_Y, num_classes=NUM_CLASSES)
        test_Y = to_categorical(sent_test_Y, num_classes=NUM_CLASSES)
        train_Y = np.array(train_Y)
        test_Y = np.array(test_Y)
        print('training shape', train_X.shape)
        print('test shape', test_X.shape)
        padded_train_X = pad_sequences(train_X, padding='post', maxlen=MAX_SEQUENCE_LEN, dtype='float')
        padded_test_X = pad_sequences(test_X, padding='post', maxlen=MAX_SEQUENCE_LEN, dtype='float')
        
        results = bert_tensorflow_test(padded_train_X, padded_test_X, train_Y, test_Y)
        TP, TN, FP, FN = results[3]
        print('ELMO, 30% test size', file=outf)
        print('eval is', results[0], file=outf)
        print('eval is', results[0])
        print('custom accuracy is', results[1], file=outf)
        print('custom accuracy is', results[1])
        print('default accuracy is', results[2], 'or', 1-results[2], file=outf)
        print('default accuracy is', results[2], 'or', 1-results[2])
        print('num train is', len(padded_train_X), file=outf) 
        print('num train is', len(padded_train_X))
        print('num test is', len(padded_test_X), file=outf)
        print('num test is', len(padded_test_X))
        print('TP', TP, 'TN', TN, 'FP', FP, 'FN', FN, file=outf)
        print('TP', TP, 'TN', TN, 'FP', FP, 'FN', FN)

        if TP == 0:
            print('precision', 0, file=outf)
            print('recall', 0, file=outf)
        else:
            print('precision', TP/(TP+FP), file=outf)
            print('recall', TP/(TP+FN), file=outf)
        break
        

    
