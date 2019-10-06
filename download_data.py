import os
import tarfile
from urllib.error import HTTPError
import urllib.request

if __name__ == "__main__":
    data_url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    dir_name = "data"
    filename = data_url.split("/")[-1]

    # Check if directory exists
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    try:
        print(f"Moving to '{dir_name}'...")
        os.chdir(dir_name)

        with urllib.request.urlopen(data_url) as response:
            with open(filename, "wb") as out_file:
                print(f"Downloading: '{data_url}' this may take several minutes.")
                data = response.read()
                out_file.write(data)
        print(f"Download finished")

        print(f"Uncompressing file: '{filename}' in '{dir_name}'")
        tar = tarfile.open(filename)
        tar.extractall()
        tar.close()
    except HTTPError as e:
        print(f"Error downloading data: {e}")
    except:
        print("Undefined error")
    finally:
        os.chdir("..")
