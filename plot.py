import numpy as np
import pandas as pd
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Specify variables which shall be used as input for the neural network
variables = ["Tau_pt"]

if __name__ == "__main__":
    # Read NanoAOD files
    def load_tree(filename):
        tree = uproot.open(filename)["Events"]
        data = {}
        for name in variables:
            if "Tau_" in name or "SV_" in name or "Jet_" in name:
                data[name] = [x[0] if len(x) > 0 else -10 for x in tree.array(name).tolist()]
            else:
                data[name] = tree.array(name)
        return pd.DataFrame(data)

    signal = load_tree("htt_nano.root")
    background = load_tree("qcd_nano.root")

    # Skim dataset
    def skim(df):
        return df[df["Tau_pt"] > 0]

    signal = skim(signal)
    background = skim(background)

    # Plot each variable
    for name in variables:
        plt.figure(figsize=(6,6))
        lims = np.percentile(signal[name], [5, 95])
        bins = np.linspace(lims[0], lims[1], 30)
        for df, label in [[signal, "Signal"], [background, "Background"]]:
            plt.hist(df[name], bins=bins, lw=3, histtype="step", label=label, density=True)
        plt.xlabel(name)
        plt.legend()
        plt.xlim((bins[0], bins[-1]))
        plt.tight_layout()
        plt.savefig(name + ".png")
        print "Created "+name+".png"
        plt.clf()
