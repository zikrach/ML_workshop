# Restrictions:

1. Requires OSX Yosemite 10.10.3 or above
2. Requires Microsoft Windows 10 Professional or Enterprise 64-bit
3. For other OS you can see restrictions on https://www.docker.com/community-edition#/download
4. Memory: 8Gb (minimum free 4Gb)
5. CPU: minimum 4 cores
5. Space: free 10Gb on disc

# I. Instructions to run ML_Game_Scenario Jupyter Notebook

1. Install Docker from https://www.docker.com/community-edition#/download
2. Run command
```
docker pull zikrach/ml_workshop:base_kiev
```
3. Run docker container (you can change munber of cores --cpus="4" and amound of RAM --memory="4g"):
```
docker run --rm --cpus="4" --memory="4g" -p 8896:8896 zikrach/ml_workshop:base_kiev /bin/bash -c "/usr/bin/run-jupyter.sh"
```
5. Open your favorite browser and navigate to the
```
http://localhost:8896/notebooks/ML_workshop/ML_Game_Scenario.ipynb
```


# II. Instruction for those who have installed the Anaconda or have an environment with the necessary libraries (scikit-learn, pandas, numpy, scipy, jupyter, and other).

1. Clone the GitHub repository:
```
git clone https://github.com/zikrach/ML_workshop.git
```
2. Open the folder with code:
```
cd ML_workshop
```
3. Run Jupyter Notebook:
```
jupyter notebook --port=8896 --no-browser --NotebookApp.token=''
```
4. Open your favorite browser and navigate to the
```
http://localhost:8896/notebooks/ML_Game_Scenario.ipynb
```


# III. Instructions to run ML_Carts Shiny application
1. Install Docker from https://www.docker.com/community-edition#/download
2. Run command
```
docker pull zikrach/ml_workshop:base_kiev
```
3. Run docker container (you can change munber of cores --cpus="4" and amound of RAM --memory="4g"):
```
docker run --rm --cpus="4" --memory="4g" -p 3838:3838 zikrach/ml_workshop:base_kiev
```
4. Open your favorite browser and navigate to the
```
http://localhost:3838/ml_carts
```
