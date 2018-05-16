#!/bin/sh

# Make sure the directory for individual app logs exists
#!/bin/bash

folder="/home/src/ML_workshop"
mkdir /home/src
cd /home/src

if [ -e $folder ]
then
 rm -rf $folder
fi

git clone https://github.com/zikrach/ML_workshop.git
cd /home/src/ML_workshop
#git checkout Kyiv_Data_Spring_2018

exec /opt/conda/bin/jupyter notebook --notebook-dir=/home/src/ --ip='*' --port=8896 --no-browser --allow-root --NotebookApp.token='' 2>&1
