## RSNA Screening Mammography Breast Cancer Detection

This is the source code for the 4th place solution to the [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/).  
Video overview: [link](TBD)  
Slides : [link](TBD)  
Submission code : [link](https://www.kaggle.com/code/darraghdog/4th-place-submission/notebook)  
Our solution write up can be found [here](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391208).
  
Sponsored by [RSNA](https://www.rsna.org/)
   

### Environment set up

For convenience of monitoring models we are using [neptune.ai](https://neptune.ai/home). At the time of writing neptune monitoring is free for researchers.
In the configs, in the `config/` directory, I have set the neptune project to `watercooled/rsna-screening`. 
You can switch this to your own neptune project, or else create a neptune project with that name.
   
We ran everthing in a nvidia docker environment. Make sure you have at least 40GB GPU memory. If you have less memory checkpoiting is set in the training configs so it will probably be fine. To set up the docker environment, 
``` 
cd ~
docker run -itd --name RSNAMAMMOGRAM -v $PWD:/mount --shm-size=1024G --gpus '"device=0"' --rm nvcr.io/nvidia/pytorch:22.10-py3
docker attach RSNAMAMMOGRAM
cd mount/
git clone https://github.com/darraghdog/RSNA-Breast-Cancer-Detection
cd RSNA-Breast-Cancer-Detection
pip install -r requirements.txt
pip install git+https://github.com/rwightman/pytorch-image-models
```

### Data Prep and training

In order to prepare the training data and folders, run `./bin/run_01_data_prep.sh`. 

In order to run the first and second stage models, rin `./bin/run_02_train.sh`

### Upload weights

You can upload the weights to a kaggle dataset with the kaggle api. Note, you only need the `*_agg1` configs uploaded. 
```
kaggle datasets init -p weights/
# Put in a name for your dataset
kaggle datasets create -r zip -p weights/
```
