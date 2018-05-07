# Restrictions:

1. Requires OSX Yosemite 10.10.3 or above
2. Requires Microsoft Windows 10 Professional or Enterprise 64-bit
3. For other OS you can see restrictions on https://www.docker.com/community-edition#/download
4. Memory: 8Gb (minimum free 4Gb)
5. CPU: minimum 4 cores
5. Space: free 10Gb on disc

# Instructions

## I. Instruction for those who want to build the image.
1. Make the folder and download file:
```
mkdir ML_workshop
cd ML_workshop
wget https://github.com/zikrach/ML_workshop/blob/master/Dockerfile
```
2. Install Docker from https://www.docker.com/community-edition#/download
3. Build docker image:
```
docker build -t ml_workshop:kyiv_2018 .
```
4. Run docker image:
```
docker run --rm --cpus="4" --memory="4g" -p 8896:8896 ml_workshop:kyiv_2018 /bin/bash -c "cd /home && git clone https://github.com/zikrach/ML_workshop.git && cd ML_workshop/ && /opt/conda/bin/jupyter notebook --notebook-dir=/home/ML_workshop --ip='*' --port=8896 --no-browser --allow-root --NotebookApp.token=''"
```
5. Open your favorite browser and navigate to the
```
http://localhost:8896/notebooks/ML_Game_Scenario.ipynb
```

## II. instruction for those who want to use an image was built by us.
1. Download files from TBD
2. Install Docker from https://www.docker.com/community-edition#/download
3. Run in terminal in folder with ml_workshop.tar file:
```
docker load -i ml_workshop.tar
```
4. Run:
```
docker run --rm --cpus="4" --memory="4g" -p 8896:8896 ml_workshop:kyiv_2018 /bin/bash -c "cd /home && git clone https://github.com/zikrach/ML_workshop.git && cd ML_workshop/ && /opt/conda/bin/jupyter notebook --notebook-dir=/home/ML_workshop --ip='*' --port=8896 --no-browser --allow-root --NotebookApp.token=''"
```
5. Open your favorite browser and navigate to the
```
http://localhost:8896/notebooks/ML_Game_Scenario.ipynb
```

### III. instruction for those who have installed the Anaconda or have an environment with the necessary libraries (scikit-learn, pandas, numpy, scipy, jupyter, and other).

1. Clone the GitHub repository:
```
git clone https://github.com/zikrach/ML_workshop.git
```
2. Open the folder with code:
```
cd ML_workshop
```
3. Run Jupyter Notebook:
4. Run:
```
jupyter notebook --port=8896 --no-browser --NotebookApp.token=''
```
5. Open your favorite browser and navigate to the
```
http://localhost:8896/notebooks/ML_Game_Scenario.ipynb
```
