from __future__ import print_function

__copyright__ = \
    """
Copyright &copyright © (c) 2020 The Board of xx University.
All rights reserved.
This software is covered by China patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.
"""
__authors__ = ""
__version__ = "1.0.0"

import os
import time
from obs import ObsClient

bucketName = 'll-coco'


def scan(local_dir, remote_dir):
    exist_set = set()
    exist_set.add(remote_dir)
    for name in os.listdir(local_dir):
        exist_set.add(os.path.join(remote_dir, name))
    return exist_set


def scan_remote(obsClient, remote_dir):
    exist_set = set()
    marker = None
    is_truncated = True
    while is_truncated:
        resp = obsClient.listObjects(bucketName, delimiter='/', prefix=remote_dir, marker=marker)
        if resp.status < 300:
            for content in resp.body.contents:
                if content.key not in exist_set:
                    exist_set.add(content.key)
            marker = resp.body.next_marker
            is_truncated = resp.body.is_truncated
        else:
            print('errorCode:', resp.errorCode)
            print('errorMessage:', resp.errorMessage)
    return exist_set


def download(obsClient, local_dir, remote_dir, delete_flag):
    exist_set = scan(local_dir, remote_dir)
    marker = None

    while True:
        resp = obsClient.listObjects(bucketName, delimiter='/', prefix=remote_dir, marker=marker)
        if resp.status < 300:
            for content in resp.body.contents:
                time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                if content.key not in exist_set:
                    key = content.key.replace(remote_dir, local_dir)
                    obsClient.getObject(bucketName, content.key, downloadPath=key)
                    print("[{}]  download:{}".format(time_str, content.key))
                elif delete_flag:
                    obsClient.deleteObject(bucketName, content.key)
                    print("[{}]  delete:{}".format(time_str, content.key))
            if len(resp.body.contents) > 0:
                marker = resp.body.contents[-1].key
        else:
            print('errorCode:', resp.errorCode)
            print('errorMessage:', resp.errorMessage)
        time.sleep(150)


def upload(obsClient, local_dirs, remote_dir, cover_flag):
    if type(local_dirs) == str:
        local_dirs = [local_dirs]
    exist_set = scan_remote(obsClient, remote_dir)
    for local_dir in local_dirs:
        for name in os.listdir(local_dir):
            local_name = os.path.join(local_dir, name)
            remote_name = os.path.join(remote_dir, name)
            if cover_flag or remote_name not in exist_set:
                resp = obsClient.putFile(bucketName, remote_name, file_path=local_name)
                if resp.status < 300:
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    print("[{}]  upload:{}".format(time_str, remote_name))
                else:
                    print('errorCode:', resp.errorCode)
                    print('errorMessage:', resp.errorMessage)


if __name__ == "__main__":
    obsClient = ObsClient(
        access_key_id='8PCTQLUHQESAUNGGOUZY',
        secret_access_key='yzTHk0D5TWgiEorKaVrBzuaEjBGDibDj9bjZoNPH',
        server='https://obs.cn-north-4.myhuaweicloud.com'
    )
    local_dir = r'C:/data/checkpoints/logs/'
    remote_dir = r'checkpoints/logs/'

    download(obsClient, local_dir, remote_dir, False)

    obsClient.close()