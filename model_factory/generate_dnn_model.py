# Keras model + getWeights(for FedAvg)
# desire functions: train, infer, get_weights, set_weights
# since we cannot set_weights or TFLite (as to date) we overcome this via model compilation of our custom weights

import os
import numpy as np
import tensorflow as tf
import sys

PWD = os.getcwd()
N_INPUTS = int(sys.argv[1]) if len(sys.argv) == 5 else int(sys.argv[2]) if len(sys.argv) == 6 else 0 # first write n_inputs global var cause decorators assing values at runtime

# external set_weights function
def set_weights(path_to_data,config):
    print("################################## LOAD CUSTOM WEIGHTS ###############################") # debug
    # read weight data
    with open(path_to_data) as f:
        lines = f.readlines() # (kernel,bias),.....,(kernel,bias)

    # convert weights to numpy array : then pass it to tf.model configuration
    weights_numpy_list = []
    counter = 0
    for key,value in config.items():
        weights = lines[counter].split('\t')[:-1] # strip new line
        weights = [float(num) for num in weights] # convert str to float
        #weights = [float(-1.11) for num in weights] # debug
        weights_numpy_list.append(np.array(weights).reshape(value).astype("float32"))
        #temp = np.array(weights).reshape(value).astype("float32")
        counter += 1

    return weights_numpy_list


class TransferLearningModel(tf.Module):
    """TF Transfer Learning model class."""

    def __init__(self, learning_rate, n_inputs, n_deep_layer, n_outputs, init):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs 
        self.n_deep_layer = n_deep_layer 
        self.n_outputs = n_outputs
        self.init = init
        
        # 1. model definition
        self.inputs = tf.keras.Input(shape=(self.n_inputs,), name="input")
        self.layer1 = tf.keras.layers.Dense(self.n_deep_layer, activation="relu",name="layer1")(self.inputs)
        self.layer2 = tf.keras.layers.Dense(self.n_deep_layer, activation="relu",name="layer2")(self.layer1)
        self.outputs = tf.keras.layers.Dense(self.n_outputs, name="layer3")(self.layer2)
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        # 2. create a list of trainable weights,layers for set_weights,get_weights
        self.trainable_weights = [weight for weight in self.model.trainable_weights]
        self.trainable_layers = [layer for layer in self.model.layers][1::] # skip first Input layer

        # 3. load suitable weights : init are the deafult weights
        if self.init == False:
            # generate model config (naive since we initially have only 2 hidden layers) (we may only want this for set weights)
            model_config = {}
            model_config['k1'] = [self.n_inputs,self.n_deep_layer]
            model_config['b1'] = [self.n_deep_layer]
            model_config['k2'] = [self.n_deep_layer,self.n_deep_layer]
            model_config['b2'] = [self.n_deep_layer]
            model_config['k3'] = [self.n_deep_layer,self.n_outputs]
            model_config['b3'] = [self.n_outputs]

            weights_type = "dnn.txt" #"wrong_dnn.txt"
            path_to_weights = os.path.join(PWD,"weights",weights_type)
            print(f"path to weights: {path_to_weights}")
            weights_numpy_list = set_weights(path_to_weights,model_config)
            for idx,(kernel,bias) in enumerate(zip(*[iter(weights_numpy_list)]*2)):
                #print("mphka")
                target_layer = self.trainable_layers[idx]
                target_layer.set_weights([kernel, bias])

        # 4. loss function and optimizer
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # no need for one-hot encoding
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


    @tf.function(input_signature=[
        tf.TensorSpec([None, N_INPUTS], tf.float32),
        tf.TensorSpec([None,], tf.float32),
    ])
    def train(self, feature, label):
        with tf.GradientTape() as tape:
            # forward pass
            logits = self.model(feature)
            loss = self.loss_fn(label,logits)

        # backprop
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return {'loss': loss}

    @tf.function(input_signature=[
        tf.TensorSpec([None, N_INPUTS], tf.float32)
    ])
    def infer(self, feature):
        logits = self.model(feature)
        probs = tf.nn.softmax(logits)
        return {'output': probs}

    # put dummy_load because tflite cannot function with None input
    @tf.function(input_signature=[tf.TensorSpec([1], tf.float32)])

    def get_weights(self, dummy_load):
        # recursively add weights to dict
        weight_dict = {}
        for weight in self.trainable_weights:
            # trim ":0" from tf name convesion
            weight_dict[weight.name[0:-2]] = weight
        return weight_dict
    

def convert_and_save(learning_rate, n_inputs, n_deep_layer, n_outputs, init):
    saved_model_dir='saved_model'
    #print(f"lr: {learning_rate}, n_inputs: {n_inputs}, n_deep_layer: {n_deep_layer}, n_outputs: {n_outputs}, init: {init}")
    model = TransferLearningModel(learning_rate,n_inputs,n_deep_layer,n_outputs,init)
    weights = model.trainable_weights
    print(f"model weights: {weights}")

    tf.saved_model.save(
        model,
        saved_model_dir,
        signatures={
            'train': model.train.get_concrete_function(),
            'infer': model.infer.get_concrete_function(),
            'get_weights': model.get_weights.get_concrete_function(),
        })
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    filename = "dnn.tflite"
    model_file_path = os.path.join(PWD,"models",filename)
    print("model_file_path: " + model_file_path)
    with open(model_file_path, 'wb') as model_file:
        model_file.write(tflite_model)


if __name__ == '__main__':
    # catch variables via command line arguments
    if len(sys.argv) == 5: # skip learning rate
        lr = 0.01
        learning_rate = lr
        n_inputs = int(sys.argv[1])
        n_deep_layer = int(sys.argv[2])
        n_outputs = int(sys.argv[3])
        #init = bool(sys.argv[4])
        init = False
        if (int(sys.argv[4]) == 1):
            init = True
        print(f"init: {init}")

    elif len(sys.argv) == 6:
        learning_rate = float(sys.argv[1])
        n_inputs = int(sys.argv[2])
        n_deep_layer = int(sys.argv[3])
        n_outputs = int(sys.argv[4])
        init = False
        if (int(sys.argv[4]) == 1):
            init = True
        print(f"init: {init}")

    else:
        print("Helper: --learning_rate: float (default=0.01)  --n_inputs:int  --n_deep_layer: int  --n_outputs: int  init: bool (compile with init weights)")
        raise SystemExit(2)

    # start compilation
    convert_and_save(learning_rate,n_inputs,n_deep_layer,n_outputs,init)
