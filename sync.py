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
import re
from obs import ObsClient
from matplotlib import pyplot as plt


bucketName = 'hyy-coco'


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
        resp = obsClient.listObjects(bucketName, delimiter='', prefix=remote_dir, marker=marker)
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


def download(obsClient, local_dir, remote_dir, delete_flag, wait):
    exist_set = scan(local_dir, remote_dir)
    marker = None
    next_flag = True

    while next_flag or wait:
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
                next_flag = True
                marker = resp.body.contents[-1].key
            else:
                next_flag = False
                if wait:
                    time.sleep(150)
        else:
            print('errorCode:', resp.errorCode)
            print('errorMessage:', resp.errorMessage)
    print("download complete!")


def upload(obsClient, local_dir, remote_dir, cover_flag, extension):
    exist_set = scan_remote(obsClient, remote_dir)
    for dp, dn, fn in os.walk(os.path.expanduser(local_dir)):
        for f in fn:
            if not f.endswith(extension):
                continue
            local_name = os.path.join(dp, f)
            remote_name = local_name.replace(local_dir, remote_dir).replace('\\', '/')
            if cover_flag or remote_name not in exist_set:
                resp = obsClient.putFile(bucketName, remote_name, file_path=local_name)
                if resp.status < 300:
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                    print("[{}]  upload:{}".format(time_str, remote_name))
                else:
                    print('errorCode:', resp.errorCode)
                    print('errorMessage:', resp.errorMessage)


def collect_loss(log_dir):
    real_pattern = r"(\d+\.\d+)"
    pattern = r"train : \[(\d+)/\d+\]\|cls_pos {0} \| cls_neg {0} \| kp_pos {0} \| kp_neg {0} " \
              r"\| ae_loss {0} \| wh {0} \| total_loss {0} \|".format(real_pattern)
    loss_list = []
    for file_name in os.listdir(log_dir):
        if 'stdout' in file_name:
            with open(os.path.join(log_dir, file_name), 'r') as f:
                lines = f.readlines()
            for line in lines:
                match_result = re.search(pattern, line)
                if match_result is not None:
                    temp = tuple([eval(e) for e in match_result.groups()])
                    loss_list.append(temp)
    loss_list.sort(key=lambda e: e[0])

    loss_tuple = tuple(zip(*loss_list))
    names = ["epoch", "cls_pos", "cls_neg", "kp_pos", "kp_neg", "ae_loss", "wh", "total_loss"]
    for index in range(1, len(names)):
        x = loss_tuple[0]
        y = loss_tuple[index]

        x_name = names[0]
        y_name = names[index]

        plt.figure(index)
        plt.plot(x, y, '-r', label=y_name)
        plt.title('Graph of {}'.format(y_name))
        plt.xlabel(x_name, color='#1C2833')
        plt.ylabel(y_name, color='#1C2833')
        plt.show()

    for index in range(len(names)):
        print(names[index], loss_tuple[index])


if __name__ == "__main__":
    obsClient = ObsClient(
        access_key_id='CKDX7ZZDE1EFF2RBKQUV',
        secret_access_key='E6fYmLUeT8jvLd5FPs9RKWKhXONv2K8ywlQKQDtd',
        server='https://obs.cn-north-4.myhuaweicloud.com'
    )
    # local_dir = r'D:\checkpoints\efficient\logs\\'
    # remote_dir = r'efficient3/logs/'
    #
    # download(obsClient, local_dir, remote_dir, False, False)

    local_dir = r"D:\cityscapes\gtFine\train\\"
    remote_dir = r"datasets/cityscapes/gtFine/train/"
    upload(obsClient, local_dir, remote_dir, False, "gtFine_instanceIds.png")

    obsClient.close()
    # collect_loss(local_dir)