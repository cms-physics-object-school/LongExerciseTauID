### 2. Tau ID

**Set up code**
```bash
source /cvmfs/grid.cern.ch/emi3ui-latest/etc/profile.d/setup-ui-example.sh
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh

export SCRAM_ARCH=slc7_amd64_gcc700
cmsrel CMSSW_10_2_15
cd CMSSW_10_2_15/src
cmsenv
git cms-init
git cms-addpkg PhysicsTools/NanoAOD 
scram b -j 8

git clone https://github.com/cms-physics-object-school/LongExerciseTauID

cd LongExerciseTauID

cmsDriver.py myNanoProdMc2018 -s NANO --mc --eventcontent NANOAODSIM --datatier NANOAODSIM  --no_exec  --conditions 102X_upgrade2018_realistic_v19 --era Run2_2018,run2_nanoAOD_102Xv1 --customise_commands="process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)))"

cp /net/data_cms/cmspos/Tau/TauID/*root .

```


**Tasks:**


1. Run the nanoAOD script on the QCD and GluGluHToTauTau miniAOD files for 100 events.

2. Inspect the output files with `python ../PhysicsTools/NanoAOD/test/inspectNanoFile.py -d output_file.html output_file.root`.

3. In `nanoAOD`, not all information about the tau leptons are kept, e.g. the information of the impact parameter of the tau track with respect to the primary vertex, and its significance, is lost. We can customize the `nanoAOD` production to add this (and potentially more) parameters back. The variables to be written out in `nanoAOD` for tau leptons are defined in `../PhysicsTools/NanoAOD/python/taus_cff.py`. Add the variables `ip3d` and `ip3d_Sig` to the output file. (You can add more if you want, the variables available in a `PAT::Tau` object are defined here: https://github.com/cms-sw/cmssw/blob/CMSSW_10_2_X/DataFormats/PatCandidates/interface/Tau.h

4. Process 100 events and check that the new variables are properly written out. After you succeeded, you can copy the three root files from `/net/data_cms/cmspos/Tau/TauID/nanoAOD_files/` to your directory, in which around 50,000 events processed with the same configuration are stored. You can use these files for the following steps to save computing time for the nanoAOD conversion.

5. Find variables which you suspect have discriminating power in seperating jets misidentified as hadronic tau leptons, and genuine hadronic tau leptons. You can add the variables to `plot.py` and check in the resulting distributions which distributions differ between the two samples.

6. Choose a set of variables in which you suspect strong discrimination power. Remember that this application will be to find a general discrimination between real and fake tau leptons, not between $H\rightarrow\tau\tau$ and QCD in general! Which variables (even if discriminating) should you not use because of this?

7. Add your variables to `train.py`. Train the neural network. Apply the model to your validation data and plot the ROC curve, and calculate the area-under-curve (AUC). 

8. Draw the output scores of your classifier for the background and signal events of the validation data.

9. Find an optimal working point (i.e. value of the NN score at which you consider a tau lepton to be genuine) for your classifier. Remember that the cross section of QCD events (jet production) is orders of magnitude higher than that of processes producing genuine tau leptons. What does this impose for your working point.

10. Give the working point at which the misidentification probabiliy (false positive rate) is at most 1%. What is the efficiency you can achieve at this point?

11. Run `python analyze.py`. This will result in a plot `ranking.png` in which the input variables of the NN are ranken according to their influence on the NN output. Did you expect the variables to be ranked as they are?

12. The file `TTToSemileptonic_MiniAOD.root` contains $t\bar{t}\rightarrow \ell \tau_{h}$ events, which means they contain both many jets (from the $t\bar{t}$ pair production) as well as genuine hadronic tau leptons. We want to seperate the misidentified and true tau leptons using our classifier.

13. Process the file to `nanoAOD` while adding all variables you need as input for your NN classifier. 

14. Apply the NN classifier for the $t\bar{t}$ events and plot the ROC curve as well as the distribution of NN scores. You can gain the truth information about the reconstructed taus by using `Tau_genPartFlav==0` for misidentified taus and `Tau_genPartFlav==5` for true hadronic taus. How does the performance compare to your validation sample?

15. How can you explain the difference in performance? Hint: Plot the $p_{T}$ distribution of genuine and fake taus from the $t\bar{t}$ sample.

16. How can you recover the performance to make the classifier more applicable to $t\bar{t}$ events?

