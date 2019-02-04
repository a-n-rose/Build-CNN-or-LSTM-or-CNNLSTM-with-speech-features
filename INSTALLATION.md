## Installation and Data Download Instructions

## Prerequisites

1. Computer with CPU

2. Optional: a recording apparatus, for example headset or earphones that have a microphone.

3. Downloaded speech data (instructions below)

4. [Python 3.6](https://www.python.org/downloads/release/python-368/)

To check your version, type the following into the command-line:

Note: I only mean the text following the '\$' sign.

```
$ python3 --version
```

Other versions might work. I just can't guarantee that.

## Virtual Environment

I suggest using a virtual environment. This allows you to use any package versions without them interfering with other programs on your system.

You can set up a virtual environment different ways. One way is with Python3.6-venv.

### Python3.6-venv

To install, enter into the command line:

```
$ sudo apt install python3.6-venv
```

or for MacOS:

```
$ pip3 install virtualenv
```

In the folder where the scripts for this workshop are, write in the command-line:

```
$ python3 -m venv env
```

This will create a folder 'env'.

Then type into the command-line:

```
$ source env/bin/activate
```

and your virtual envioronment will be activated. Here you can install packages that will not interfere with any other programs.

To deactivate this environment, simply type in the command line:

```
$ deactivate
```

## Installation

To install all the python packages we will use, first start up your virtual environment:

```
$ source env/bin/activate
```

In your virtual environment, run 'requirements.txt' to install all necessary packages via pip. This should only take a couple of minutes.

```
(env)..$ pip install -r requirements.txt
```

### OPTIONAL INSTALLATION

If you also want to run 'implement_model.py', which will need to record audio, you will also need to install sounddevice and soundfile. You can do this with the following:

```
(env)..$ pip install -r requirements_implementmodel.txt
```

### POSSIBLE PROBLEMS WITH INSTALLATION:

To double check installation of matplotlib is totally fine. Open a python shell (by typing ```python```) and type:

```
import matplotlib.pyplot as plt
```

If that raises an error related to tk inter, then tk inter support is missing from your python. Check [tk_tinker](https://wiki.python.org/moin/TkInter) for installation instructions or for ubuntu you can do:

```
$ sudo apt-get install python3-tk

$ sudo apt-get install tk-dev
```
If that doesn't fix the issue, reinstalling python and setting up the environment will be needed to make python support tk inter. Check [pyenv](https://github.com/pyenv/pyenv#simple-python-version-management-pyenv) for easy python version installations.

## Download the Data

1. In the directory where the workshop scripts are, create a folder called 'data'.

2. Download the speech commands dataset <a href="http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz">here</a>. Save the zip folder in the folder named 'data'.

3. Extract the zipfile. The data directory should look like this:

![Imgur](https://i.imgur.com/fqSzLVm.png)

If instead you have a folder 'speech_commands_0.01', move the folders within that folder up one level. 

### POSSIBLE PROBLEMS WITH DOWNLOAD:

If you have trouble unzipping this file, try installing <a href="https://www.7-zip.org/">7-zip</a>. Then use that to unzip the file.

## Other Data Options

You are welcome to use other data. I have instructions <a href="https://a-n-rose.github.io/2019/01/31/small-female-male-speech-data.html">here</a> for downloading female and male speech data from an online speech database. Ensure that the contents of the 'data' folder look like the picture above (with the folder names matching the classes of the data you want to use, e.g. "male_speech" and "female_speech".
