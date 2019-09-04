import numpy as np
np.random.seed(1234)
import pandas as pd
import uproot
import pickle



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TODO: Add more variables
# Specify variables which shall be used as input for the neural network
variables = ["Tau_pt"]

if __name__ == "__main__":
    # Read NanoAOD files
    def load_tree(filename):
        tree = uproot.open(filename)["Events"]
        data = {}
        for name in variables:
            # Only use tau lepton leading in pT. Set a default (here -10) for events which has no tau lepton at all.
            if "Tau_" in name: 
                data[name] = [x[0] if len(x) > 0 else -10 for x in tree.array(name).tolist()]
            else:
                data[name] = tree.array(name)
        return pd.DataFrame(data)

    signal = load_tree("htt_nano.root")
    background = load_tree("qcd_nano.root")

    # Skim dataset
    def skim(df):
        # Remove events which have not tau lepton at all, determined by the fact that we set Tau_pt to -10 in these cases.
        return df[df["Tau_pt"] > 0]

    signal = skim(signal)
    background = skim(background)

    # Compute training weights.
    sum_entries = len(signal) + len(background)

    def reweight(df):
        # Reweight samples to the same importance - this will tell the neural net that we are equally interested in finding fake and genuine tau leptons.
        # Is this the optimal setting?
        df["training_weight"] = sum_entries / float(len(df))

    reweight(signal)
    reweight(background)

    # Load data in format understood by Keras
    def to_numpy(df, class_label):
        x = df.as_matrix(columns=variables)
        y = np.ones(len(df)) * class_label
        w = df.as_matrix(columns=["training_weight"])
        return x, y, w

    # The input variables are stored in an array "x", the desired output are stored in an array "y". In our case this is "0" for fake taus, and "1" for genuine taus.
    x_sig, y_sig, w_sig = to_numpy(signal, class_label=1)
    x_bkg, y_bkg, w_bkg = to_numpy(background, class_label=0)

    # Stack numpy arrays of different classes to single array
    x = np.vstack([x_sig, x_bkg])
    y = np.hstack([y_sig, y_bkg]).squeeze()
    w = np.vstack([w_sig, w_bkg]).squeeze()

    print ("Input variables: ",x)
    print ("True output class: ",y)
    print ("Training weights: ",w)

    # Train preprocessing and transform inputs
    # This will transform variable distributions to the same mean and variance
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    # TODO: Create new Sequential neural network called "model" (see tutorial on slides or https://keras.io/)
    # TODO: Add one dense hidden layer (around 100 neurons), and one dense output layer (1 neuron). Use a "relu" activation function for the hidden layer, and a "sigmoid" activation function for the output layer.
    # model = 
    # TODO: Specify loss function and optimizer algorithm
    
    # Print architecture
    model.summary()

    # Split dataset in training and validation part used for the gradient steps and monitoring of the training
    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(x, y, w, test_size=0.5, random_state=1234)

    # TODO: Declare callbacks to be used during training (keras.callbacks.ModelCheckpoint and keras.callbacks.EarlyStopping)
    # Early stopping will stop training if no improvement was achieved the last N epochs (N is set by "patience" parameter)
    
    # TODO: Train the neural network

    # Plot loss on training and validation dataset
    plt.figure(figsize=(6,6))
    epochs = range(1, len(history.history["loss"])+1)
    plt.plot(epochs, history.history["loss"], "o-", label="Training")
    plt.plot(epochs, history.history["val_loss"], "o-", label="Validation")
    plt.xlabel("Epochs"), plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.png")

    # TODO
    # Make prediction on validation dataset and compute ROC and AUC
  
    # TODO
    # Get NN scores for true signal and true background events

    # TODO
    # Plot NN output

    # Serialize the inputs for the analysis of the gradients
    pickle.dump(x, open("x.pickle", "wb"))
