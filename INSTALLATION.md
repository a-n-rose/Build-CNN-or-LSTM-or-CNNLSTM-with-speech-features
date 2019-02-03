
## Installation and Data Download Instructions

 
## Prerequisites

1) Computer with CPU 

2) Optional: a recording apparatus, for example headset or earphones that have a microphone.

3) Downloaded speech data (instructions below)

4) Python 3.6

To check your version, type the following into the command-line:

Note: I only mean the text following the '$' sign. 

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

That should be all you need. But...

If you need to install tk_tinker as well, here is what you can do:

```
$ sudo apt-get install python3-tk

$ sudo apt-get install tk-dev
```

## Download the Data

1) In the directory where the workshop scripts are, create a folder called 'data'.

2) Download the speech commands dataset <a href="http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz">here</a>. Save the zip folder in the folder named 'data'.

3) Extract the zipfile.

