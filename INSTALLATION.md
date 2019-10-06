# Installation and Data Download Instructions

## Prerequisites

1. Computer with a good CPU,
2. (Optional) A recording apparatus, for example headset or earphones
   that have a microphone,
3. Downloaded speech data (instructions below),
4. [Python >= 3.6](https://www.python.org/downloads/).

To check your version, type the following into a terminal:
```
python --version
```
in some systems `python` is called `python3`, so you can also try:
```
python3 --version
```
Other versions might work. I just can't guarantee that.

Hereon, remember which one worked for you, `python` or `python3`
and use it for all the snippets you see in the instructions.

## Virtual Environment

I suggest using a virtual environment.
This allows you to use any package versions without them interfering with
other programs on your system.

You can set up a virtual environment different ways.
Two ways are with the `venv` or `virtualenv` module.

To check if they are installed, you can run on a terminal the following
command:
```
python -c "import venv"
```
and
```
python -c "import virtualenv"
```

If one of them worked, you can use it to create a virtual environment,
otherwise you need to install one:

### Installing `venv`
Windows and macOS:
* `pip install venv` (maybe `pip3`)
Linux:
* Besides using `pip`, you can install this using your OS package manager:
 * `apt-get install python3.6-venv` (Ubuntu)

### Installing `virtualenv`

Same procedures are with `venv` but using `virtualenv` instead.

### Creating environment

In the folder where the scripts for this workshop are, write in the command-line:

```
python -m venv env
```
or
```
virtualenv env
```

This will create a folder 'env'.

### Activating the environment

Simply type the following:
```
source env/bin/activate
```
and your virtual environment will be activated.

(You will be able to deactivate it by running `deactivate` on the same terminal)

Here you can install packages that will not interfere with any other programs.

## Installation

To install all the python packages we will use,
remember to activate your virtual environment:
```
source env/bin/activate
```

In your virtual environment, use `requirements.txt` to install all necessary
packages via `pip`. This should only take a couple of minutes.
```
pip install -r requirements.txt
```

## (Optional) Installation

If you also want to run `implement_model.py`, which will need to record audio,
you will also need to install `sounddevice` and `soundfile`.
You can do this with the following:
```
pip install -r requirements_implementmodel.txt
```

## Possible installation issues

* To double check the installation of `matplotlib` execute the following
  command, and you should not receive any error:

  ```
  python -c "import matplotlib"
  ```

  If that raises an error related to `tk` inter, then its support is missing
  from your python.
  Check [tk_tinker](https://wiki.python.org/moin/TkInter) for installation
  instructions or for Ubuntu you can do:
  ```
  sudo apt-get install python3-tk tk-dev
  ```
  If that doesn't fix the issue, reinstalling python and setting up the
  environment will be needed to make python support tk inter.
  Check [pyenv](https://github.com/pyenv/pyenv#simple-python-version-management-pyenv)
  for easy python version installations.

## Data download

### Automatic

Use the `download_data.py` script to automate this process by just typing:
```
python download_data.py
```

### Manual

1. In the directory where the workshop scripts are, create a folder called 'data'.
2. Download the speech commands data set
   [here](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz).
   Save the tar file in a folder named `data`.
3. Extract the tar file. The data directory should look like this:
   ![Imgur](https://i.imgur.com/fqSzLVm.png)

If instead you have a folder `speech_commands_0.01`,
move the folders within that folder up one level.

### Possible problems with the downloaded file

If you have trouble unzipping the file,
try installing [7-zip](https://www.7-zip.org/).
Then use that to unzip the file.

## Other Data Options

You are welcome to use other data sets.
I have instructions [here](https://a-n-rose.github.io/2019/01/31/small-female-male-speech-data.html)
for downloading female and male speech data from an online speech database.
Ensure that the contents of the `data` folder look like the picture above
(with the folder names matching the classes of the data you want to use,
e.g. `male_speech` and `female_speech`).
