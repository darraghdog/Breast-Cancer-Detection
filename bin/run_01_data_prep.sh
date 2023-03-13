mkdir datamount/
mkdir weights/

# Download competition data
kaggle competitions download -c rsna-breast-cancer-detection
unzip rsna-breast-cancer-detection.zip -d datamount/
rm rsna-breast-cancer-detection.zip

# Create a train file with 5 folds included, stratified on cancer label
python make_folds_v01.py

# Window over the dicom, crop the breast area and save to png
python crop_mammograms_and_convert_to_png.py --datadir datamount --n_cores 16
