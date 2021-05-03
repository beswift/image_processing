
# A dockerfile must always start by importing the base image.
# We use the keyword 'FROM' to do that.
# In our example, we want import the python image.
# So we write 'python' for the image name and 'latest' for the version.
# base image
# a little overkill?
# FROM ubuntu:18.04
# FROM python:latest

FROM continuumio/miniconda3


# Install basic stuff
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates && \
    sudo apt install -y tesseract-ocr &&\
    sudo apt install -y libtesseract-dev &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

## Install miniconda
## ENV PATH $CONDA_DIR/bin:$PATH
## RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    #echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    #/bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    #rm -rf /tmp/*

# Speed up conda
#RUN conda install -y mamba -c conda-forge





# exposing default port for streamlit
EXPOSE 8501

# making directory of app
WORKDIR /image_alignment

# ubuntu installing - python, pip,
#RUN apt-get update &&\
    #apt-get install -y \
    #python3-pip\
    #libsm6 libxext6 libxrender-dev
# RUN apt-getpip install --upgrade pip
# copy over requirements
COPY environment_1.yml .
#RUN mamba env update --file ./environment_1.yml &&\
    #conda clean -tipy

RUN conda env create -f environment_1.yml

# COPY requirements.txt ./requirements.txt

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "img_base", "/bin/bash", "-c"]

# installing required packages from requirements.txt
# RUN pip3 install -r requirements.txt

# In order to launch our python code, we must import it into our image.
# We use the keyword 'COPY' to do that.
# The first parameter 'main.py' is the name of the file on the host.
# The second parameter '/' is the path where to put the file on the image.
# Here we put the file at the image root folder.
# copying all app files to image
# COPY main.py /
COPY . .

# cmd to launch app when container is run
# We need to define the command to launch when we are going to run the image.
# We use the keyword 'CMD' to do that.
# The following command will execute "python ./main.py".
# CMD [ "python", "./main.py" ]
# The following command will execute scripts running in a subdir
# CMD python3 scripts/scripts.py

# The code to run when container is started:
#COPY main.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "img_base"]

# just run streamlit
CMD ["streamlit", "run","main.py"]


