import sys, os, zipfile
from urllib.request import urlretrieve

import time

import datetime


class DownloadReporter:

    def __init__(self, filename):
        self.t_last = datetime.datetime.now()
        print("-> started downloading file '{}'".format(filename))

    def reporthook(self, blocknum, blocksize, totalsize):
        bytesread = blocknum * blocksize
        t_cur = datetime.datetime.now()
        t_diff = t_cur - self.t_last
        if t_diff.total_seconds() >= 1:
            print("\rdata downloaded: {0:.2f}mb".format(bytesread / 1000000.0), end='')
            self.t_last = t_cur


def unpack_zip(filename: str):
    zfile = zipfile.ZipFile(filename)
    for f in zfile.namelist():
        if f.endswith('/'):
            os.makedirs(f)
        else:
            zfile.extract(f)


names = ["CNN_RAND_80", "ConvLong", "LT2"]
print("Downloading 3 large models")
print("Download size ~ 1GB")
print("This could take some time !")
time.sleep(2)

for filename in names:
    print("")
    zfilename = filename + ".zip"
    url = "https://www.oliver-feucht.de/nextcloud/s/OrPDp7G2uxLo5av/download?path=%2F&files=" + filename + "&downloadStartSecret=9lxw8kh983"
    dhook = DownloadReporter(zfilename)
    urlretrieve(url, zfilename, dhook.reporthook)
    print("\n-> finished downloading '{}'".format(zfilename))

    print("-> starting unzipping file '{}'".format(zfilename))
    unpack_zip(zfilename)
    print("-> finished unzipping file '{}'".format(zfilename))
    os.remove(zfilename)
