import numpy as np
from sklearn.model_selection import train_test_split
import gzip as gz

inverted = {"A":"T", "T":"A", "C":"G", "G":"C"}
padding = [0.0,0,0,0]
ix = ["A", "C", "G","T"]
one_hot_conv = {"A": [1,0,0,0],
           "T": [0,0,0,1],
           "C": [0,1,0,0],
           "G": [0,0,1,0]}

def reverse_complement(sequence):
    return [inverted[base] for base in reversed(sequence)]

def one_hot(sequence):
    return np.array([np.array(one_hot_conv[base], dtype=np.float) for base in sequence] )
    
from keras.models import model_from_json, model_from_yaml
import yaml

def save_model(model, model_name):
    with open(model_name+".json", 'w') as j_file:
        j_file.write(model.to_json())

    model.save_weights(model_name+".h5")
    
def save_model_yaml(model, model_name):
    with open(model_name+".yaml", 'w') as j_file:
        j_file.write(model.to_yaml())

    model.save_weights(model_name+".h5")
    
def load_model(model_name):
    
    with open(model_name + ".json", 'r') as j_file:
        loaded_model_json = j_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name + ".h5")
    
    return loaded_model

def load_model_yaml(model_name, cust_objects={}):
    with open(model_name + ".yaml", 'r') as y_file:
        loaded_model_yaml = y_file.read()
    
    loaded_model = model_from_yaml(loaded_model_yaml,custom_objects=cust_objects)
    loaded_model.load_weights(model_name + ".h5")
    
    return loaded_model

def train_test_val_split(x, y, rs=1):

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=rs)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=rs)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_fasta_gz(f_name):
    sequences = []
    cur_string = ""
    s = 0
    with gz.open(f_name) as fasta_file:
        for line in fasta_file:
            line = line.decode("ascii")
            if line[0] == '>':
                s+=1
                if cur_string:
                    assert len(cur_string) ==1000
                    sequences.append(cur_string)

                cur_string = ""
            else:
                line = line.strip()
                cur_string += line

        assert len(cur_string) ==1000
        sequences.append(cur_string)


    return sequences


def load_recomb_data(aug):
    hot_sequences = load_fasta_gz("hotspots.fasta.gz")
    cold_sequences = load_fasta_gz("coldspots.fasta.gz")
    if aug:
        X = np.array([one_hot(seq) for seq in hot_sequences] + [one_hot(reverse_complement(seq)) for seq in hot_sequences] + [one_hot(seq) for seq in cold_sequences] + [one_hot(reverse_complement(seq)) for seq in cold_sequences])[:,:997] # to make the pooling symmetric 
        Y = np.array([1]*(2*len(hot_sequences)) + [0]*(2*len(cold_sequences)))
    else:
        X = np.array([one_hot(seq) for seq in hot_sequences] + [one_hot(seq) for seq in cold_sequences])[:,:997] # to make the pooling symmetric 
        Y = np.array([1]*len(hot_sequences) + [0]*len(cold_sequences))

    X_train, X_val, X_test, Y_train, Y_val, Y_test = train_test_val_split(X, Y)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
