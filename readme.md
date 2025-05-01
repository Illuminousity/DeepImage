
# DeepImage - A Quantitative Analysis of CNN Approaches to Image Reconstruction through Disordered Media

> This repository outlines all the code used in the honours project: **A Quantitative Analysis of CNN Approaches to Image Reconstruction through Disordered Media** by **Matthew Robertson** (**52199806**)

## Install Prerequisites
**GENERAL**
* Install `requirements.txt` using `pip install -r requirements.txt`

**FOR DATA COLLECTION**

* Install Thorcam - Installer found in `./Installers`
* Install ALP4.3 Libraries - Installer found in `./Installers`
* Install Thorcam Camera Libs to Python - Instructions found in Thorlabs Repo https://github.com/Thorlabs/Camera_Examples/tree/main/Python#build-instructions
* After Installing `requirements.txt` using `pip` the folder located at `C:\Program Files\ALP-4.3\ALP-4.3 API` is renamed to `C:\Program Files\ALP-4.3\ALP-4.3 high-speed API` this is an issue with ALP4Lib referencing an outdated folder.

## Collecting Data

1. Firstly the CMOS Camera will need calibrated, ensure all hardware (Camera & DMD) are connected to the PC and turned on.
2. Run `./Data Collection/dmd_test.py` this will project an image on the DMD for calibration
3. Open Thorcam which should now be installed, Ensure the camera is detected and select that camera.
4. You should now be able to click the 'Play' button to preview what the camera can see, to load the ROI options click on the 'Cog' icon and click 'Load ROI Settings' and select one of the `CS165MU.json` files, finally confirm that the camera can see the image.
5. Now we have calibrated the setup, halt `./Data Collection/dmd_test.py` and run one of the `./Data Collection/collect_data.py` depending on if you are collecting greyscale images or not.
6. You should now be able to leave the script to run, you can adjust how many EMNIST images will be iterated in the `./Data Collection/collect_data.py` python file using the main for loop.

## Training Models

`Argparse` has been setup to make models easy to train, especially because of the 224 models that have been trained over the course of the project.

To train a model the following example command can be used `./NNArchitectures/TrainModel.py --grit_value '600' --cap '60000' --lr 1e-4 --epochs 20 --batchsz 4 --architecture 'resnet_unet' --loss 'l1'` This refers to a model being trained at a 600 GRIT on 60,000 images with an initial learning rate of 1e-4, over 20 epochs, a batchsize of 4, on the ResNetUNet architecture and on a L1 loss function. All of these parameters are modifyable given that the data exists.

## Testing Models

After a Model has been trained it will be saved to the root directory of the repository, a quick test on a single trained model can be ran using `./NNArchitectures/TrainModel.py` this file also uses `argparse` and a model can be loaded using `./NNArchitectures/TestModel.py --grit 1500 --model_path 'effnet_unet_1500_60000_l1_greyscale.pth' --greyscale --architecture 'effnet_unet'`

An Evaluation of all models in the root directory can be done using `./Eval/EvalModels.py` the data collected from this evaluation will be stored in a csv file in the `./csveval` folder

## Utils

The `./Utils` folder contains scripts that have some sort of Utility function, moving files fast ("This is an issue for Windows Explorer when dealing with 600,000 images!"), emulating scattering/gaussian blur effects, many of the files in Utils went unused in the main project but are reminiscent of the starting points to the project.

## Eval

The `./Eval` folder contains scripts that have been used to generate figures or statistics within the dissertation, including ANOVA, feature map visualisation and others.

## Data

The `./Figures`, `./FiguresGreyscale`, `./csv`, and `./csveval` folders contain either matplotlib figures of training results or contain csv files that record the actual numbers for these figures. The `./csveval` folder specifically contains the testing results from all of the models considered in the project.

## Not Included due to Size

The `./DMD` and `./data` folders are too large to upload to the GitHub Repo, and shouldn't be on the GitHub Repo. They contain images used within the training process, the `./data` folder is the EMNIST data from EMNIST directly that is not in an image format. The `./DMD/` folder contains all of the images that were taken throughout the project over both greyscale and binary images and over the ranges of diffusion, both training and testing datasets or contained within this folder. There is over 6GB of data collected over the course of the project.

## ZIP Submission

Due to size limitations for the ZIP submission, a variety of models have been picked and a small sample of testing data has been supplied.