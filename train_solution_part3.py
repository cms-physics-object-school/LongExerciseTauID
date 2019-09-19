import numpy as np
np.random.seed(1234)
import pandas as pd
import uproot
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This is a possible variable selection. Some of these variables (Tau_ip3d, Tau_ip3d_Sig) are only available in nanoAOD if the nanoAOD conversion is customized. 
# Specify variables which shall be used as input for the neural network
variables = ["Tau_flightLength", "Tau_flightLengthSig", "Tau_ip3d", "Tau_ip3d_Sig", "Tau_chargedIso",  "Tau_decayMode", "Tau_neutralIso", 
             "Tau_dxy", "Tau_eta","Tau_dz", "Tau_photonsOutsideSignalCone", "Tau_mass", "Tau_leadTkPtOverTauPt", "Tau_leadTkDeltaEta", "Tau_leadTkDeltaPhi","Tau_pt"]

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

    # SOLUTION part 3: One way to improve classifier
    # Reweight pT distribution of input samples to restore performance for ttbar - and make classifier pT-independent
    bins = np.array([0,21,22,23,24,25,26,27,28,29,30,32,34,36,38,40,44,48,52,56,60,64,68,74,78,85,90,100,200,1000000])
    ptSig, binEdges = np.histogram(signal["Tau_pt"], bins=bins, density=True)
    ptBkg, binEdges = np.histogram(background["Tau_pt"], bins=bins, density=True)
    weights = np.divide(ptBkg,ptSig)

    ids = np.digitize(background["Tau_pt"].values,bins)

    for i in range(len(background["training_weight"].values)):
        background["training_weight"].values[i] = background["training_weight"].values[i] / weights[ids[i]-1] 

    plt.figure(figsize=(6,6))
    lims = np.percentile(signal["Tau_pt"], [5, 95])
    bins = np.linspace(lims[0], lims[1], 30)
    plt.hist([signal["Tau_pt"],background["Tau_pt"]], weights = [signal["training_weight"],background["training_weight"]], bins=bins, lw=3, histtype="step", label=["signal","background"], normed=1)
    plt.xlabel("Tau_pt")
    plt.legend()
    plt.xlim((bins[0], bins[-1]))
    plt.tight_layout()
    plt.savefig("Tau_pt_reweighted.png")
    plt.clf()

    # Load data in format understood by Keras
    def to_numpy(df, class_label, variables_for_nn):
        x = df.as_matrix(columns=variables_for_nn)
        y = np.ones(len(df)) * class_label
        w = df.as_matrix(columns=["training_weight"])
        return x, y, w

    # The input variables are stored in an array "x", the desired output are stored in an array "y". In our case this is "0" for fake taus, and "1" for genuine taus.
    x_sig, y_sig, w_sig = to_numpy(signal, class_label=1, variables_for_nn = variables)
    x_bkg, y_bkg, w_bkg = to_numpy(background, class_label=0, variables_for_nn = variables)

    # Stack numpy arrays of different classes to single array
    x = np.vstack([x_sig, x_bkg])
    y = np.hstack([y_sig, y_bkg]).squeeze()
    w = np.vstack([w_sig, w_bkg]).squeeze()

    # Train preprocessing and transform inputs
    # This will transform variable distributions to the same mean and variance
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    # Stack layers defining neural network architecture: 1 hidden layer with 10 neurons, 1 output node
    # NN model can be adapted: Number of layers, nodes, activation functions for hidden layer and output node ...
    model = Sequential()
    model.add(Dense(100, activation="relu", input_dim=len(variables)))
    model.add(Dense(1, activation="sigmoid"))

    # Specify loss function and optimizer algorithm
    # Again loss function and optimizer was chosen by us and can be adapted
    model.compile(loss="binary_crossentropy", optimizer="adam")

    # Print architecture
    model.summary()

    # Split dataset in training and validation part used for the gradient steps and monitoring of the training
    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(x, y, w, test_size=0.5, random_state=1234)

    # Declare callbacks to be used during training
    # Early stopping will stop training if no improvement was achieved the last N epochs (N is set by "patience" parameter)
    model_checkpoint = ModelCheckpoint("model.h5", save_best_only=True, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # Train the neural network
    history = model.fit(
                    x_train,
                    y_train,
                    sample_weight=w_train,
                    batch_size=100,
                    epochs=10000,
                    validation_data=(x_val, y_val, w_val),
                    callbacks=[early_stopping, model_checkpoint])

    # Plot loss on training and validation dataset
    plt.figure(figsize=(6,6))
    epochs = range(1, len(history.history["loss"])+1)
    plt.plot(epochs, history.history["loss"], "o-", label="Training")
    plt.plot(epochs, history.history["val_loss"], "o-", label="Validation")
    plt.xlabel("Epochs"), plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.png")

    # Make prediction on validation dataset and compute ROC and AUC
    y_pred = model.predict(x_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    auc_value = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label="AUC = {:.3f}".format(auc_value))
    plt.xlabel('Misidentification probability')
    plt.ylabel('Efficiency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc.png")

    # Get NN scores for true signal and true background events
    sig_pred = [score for score,truth in zip(y_pred,y_val) if truth==1]
    bkg_pred = [score for score,truth in zip(y_pred,y_val) if truth==0]

    # Plot NN output
    plt.figure(figsize=(6,6))
    plt.hist(
    [sig_pred,bkg_pred],
    normed=1,
    bins=30,
    histtype="step",
    label = ["signal", "background"],
    lw=2)
    plt.xlabel("NN score")
    plt.legend()
    plt.savefig("nn_score.png")
    plt.clf()

    # Serialize the inputs for the analysis of the gradients
    pickle.dump(x, open("x.pickle", "wb"))

    # TTbar part
    variables.append("Tau_genPartFlav")

    ttbar = load_tree("tt_semileptonic_nano.root")
    ttbar = skim(ttbar)
    ttbar_fakes = ttbar[ttbar["Tau_genPartFlav"] == 0]
    ttbar_true = ttbar[ttbar["Tau_genPartFlav"] == 5]

    # We do not train on ttbar sample yet, so no training weight needs to be specified
    ttbar_fakes["training_weight"] = 1.0
    ttbar_true["training_weight"] = 1.0

    x_tt, y_tt, w_tt = to_numpy(ttbar_fakes, class_label=0, variables_for_nn = [x for x in variables if x not in  ["Tau_genPartFlav"]])
    x_tt_true, y_tt_true, w_tt_true = to_numpy(ttbar_true, class_label=1, variables_for_nn = [x for x in variables if x not in  ["Tau_genPartFlav"]])
    x_tt = scaler.transform(x_tt)
    x_tt_true = scaler.transform(x_tt_true)

    y_ttbar_pred_fake = model.predict(x_tt)
    y_ttbar_pred_true = model.predict(x_tt_true)

    # Make prediction on validation dataset and compute ROC and AUC
    fpr, tpr, thresholds = roc_curve(np.concatenate((y_tt,y_tt_true),axis=0), np.concatenate((y_ttbar_pred_fake,y_ttbar_pred_true),axis=0))
    auc_value = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label="AUC = {:.3f}".format(auc_value))
    plt.xlabel('Misidentification probability')
    plt.ylabel('Efficiency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_tt.png")

    # Plot NN output
    plt.figure(figsize=(6,6))
    plt.hist(
    [y_ttbar_pred_fake, y_ttbar_pred_true],
    bins=30,
    histtype="step",
    normed = 1,
    label = ["Fakes", "True taus"],
    lw=2)
    plt.xlabel("NN score")
    plt.legend()
    plt.savefig("nn_score_ttbar.png")
    plt.clf()
