# https://dzone.com/articles/simple-examples-of-downloading-files-using-python
import requests
import os
import time

dir_path = os.path.join(os.getenv("HOME"), __file__ + "_" + time.strftime("%Y%m%d-%H%M%S"))

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for i in range(50):

    file_path = os.path.join(dir_path, f"000{i:02d}.jpg")

    url = f"https://ufdcimages.uflib.ufl.edu/AA/00/01/36/43/00001/000{i:02d}.jpg"

    try:
        req = requests.get(url)
        req.raise_for_status()
        open(file_path, 'wb').write(req.content)
    except requests.exceptions.HTTPError as err:
        print(err)
        # raise SystemExit(err)

    print(file_path)

# install imagemagick, call $ sudo apt-get install imagemagick
# check ghostscript version, call $ gs --version
# should be 9.24+
# https://askubuntu.com/questions/1081895/trouble-with-batch-conversion-of-png-to-pdf-using-convert

# cancatenate jpegs into pdf
# $ convert 00*.jpg book.pdf

