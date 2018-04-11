FROM continuumio/anaconda3

# Add application codebase
#RUN mkdir /home/ML_workshop
#ADD ./ /home/ML_workshop




# Install additional Ubuntu packages
RUN apt-get update && apt-get -y install apt-utils
RUN apt-get -y install libgl1-mesa-glx mc htop libssl-dev


RUN /opt/conda/bin/conda update -n base conda
RUN /opt/conda/bin/conda install jupyter -y --quiet


# Build Instructions
# 1.
#docker build -t ml_workshop:base .

# Instructions
# 1.
#docker load -i ml_workshop.tar
# 2.
#docker run --rm --cpus="4" --memory="4g" -p 8896:8896 ml_workshop:base /bin/bash -c "git clone https://github.com/zikrach/ML_workshop.git && cd ML_workshop/ && /opt/conda/bin/jupyter notebook --notebook-dir=/home/ML_workshop --ip='*' --port=8896 --no-browser --allow-root --NotebookApp.token=''"
# 3. Open in browser
#http://localhost:8896/notebooks/
