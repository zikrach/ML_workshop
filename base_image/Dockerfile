FROM r-base:latest

MAINTAINER Winston Chang "winston@rstudio.com"

# Install dependencies and Download and install shiny server
RUN apt-get update && apt-get install -y -t unstable \
    sudo \
    gdebi-core \
    pandoc \
    pandoc-citeproc \
    libcurl4-gnutls-dev \
    libcairo2-dev/unstable \
    libxt-dev && \
    wget --no-verbose https://s3.amazonaws.com/rstudio-shiny-server-os-build/ubuntu-12.04/x86_64/VERSION -O "version.txt" && \
    VERSION=$(cat version.txt)  && \
    wget --no-verbose "https://s3.amazonaws.com/rstudio-shiny-server-os-build/ubuntu-12.04/x86_64/shiny-server-$VERSION-amd64.deb" -O ss-latest.deb && \
    gdebi -n ss-latest.deb && \
    rm -f version.txt ss-latest.deb && \
    R -e "install.packages(c('shiny', 'rmarkdown'), repos='https://cran.rstudio.com/')" && \
    cp -R /usr/local/lib/R/site-library/shiny/examples/* /srv/shiny-server/ && \
    rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update && sudo apt-get -y install git wget libssl-dev nano
RUN mkdir /home/src && cd /home/src && git clone https://e275dc090336b5a85228f5767f4400ed1536f1ec@github.com/zikrach/framework-ml-carts.git

RUN mkdir /srv/shiny-server/ml_carts && cp -a /home/src/framework-ml-carts/shiny_with_classification/. /srv/shiny-server/ml_carts
RUN chmod a+rwx -R /srv/shiny-server/ml_carts


# Temporal fix for RStan package
# http://discourse.mc-stan.org/t/error-when-installing-rstan-2-16-2/1730
# https://github.com/stan-dev/rstan/wiki/Installing-RStan-on-Mac-or-Linux
RUN echo "\nCXXFLAGS=-O3 -mtune=native -march=native -Wno-unused-variable -Wno-unused-function" >> /etc/R/Makeconf
RUN echo "\nCXXFLAGS+=-DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION" >> /etc/R/Makeconf
RUN echo "\nCPPFLAGS+=-DBOOST_PHOENIX_NO_VARIADIC_EXPRESSION" >> /etc/R/Makeconf


# Install additional R packages
RUN R -e "install.packages(c('ggplot2', 'devtools','proxy','DT','caret','lattice'))"
RUN R -e "install.packages(c('plotly','e1071','dbscan','naivebayes','ranger')); devtools::install_github('Zelazny7/isofor')"
RUN R -e "install.packages(c('adabag','plyr','kernlab','kknn','RANN','RSNNS','tictoc','C50','cluster','e1071','shinyBS'))"
RUN R -e "install.packages(c('shinyjs','shinydashboard','shiny','dplyr','SDMTools'))"


EXPOSE 3838

COPY shiny-server.sh /usr/bin/shiny-server.sh

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENTRYPOINT [ "/usr/bin/tini", "--" ]

RUN apt-get update && apt-get -y install apt-utils libgl1-mesa-glx mc htop libssl-dev
RUN /opt/conda/bin/conda update -n base conda
RUN /opt/conda/bin/conda install jupyter tqdm -y --quiet


EXPOSE 8896

COPY run-jupyter.sh /usr/bin/run-jupyter.sh
RUN chmod +x /usr/bin/run-jupyter.sh
RUN chmod +x /usr/bin/shiny-server.sh


CMD ["/usr/bin/shiny-server.sh", "/usr/bin/run-jupyter.sh"]
