__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

from modelarts.session import Session
from modelarts.estimator import Estimator
import time


def create_job():
    session = Session(username='hw48658147', password='Ll812132249', region_name='cn-north-4')
    estimator = Estimator(
        modelarts_session=session,
        framework_type='PyTorch',  # AI引擎名称
        framework_version='PyTorch-1.0.0-python3.6',  # AI引擎版本
        code_dir='/ll-coco/codes/',  # 训练脚本目录
        boot_file='/ll-coco/codes/train.py',  # 训练启动脚本目录
        log_url='/ll-coco/checkpoints/txtlogs/',  # 训练日志目录
        hyperparameters=[
            {"label": "cfg_path",
             "value": "s3://ll-coco/codes/configs/train_cfg.yaml"}
        ],
        output_path='/ll-coco/checkpoints/',  # 训练输出目录
        train_instance_type='modelarts.vm.gpu.free',  # 训练环境规格
        train_instance_count=1)
    estimator.fit(inputs='/ll-coco/datasets/cityscapes/', wait=False, job_name='erf')

    print("{} job created".format(time.time()))
    time.sleep(3600)


def create_loop():
    session = Session(username='hw48658147', password='Ll812132249', region_name='cn-north-4')
    job_list_info = Estimator.get_job_list(modelarts_session=session, per_page=10, page=1, order="asc",
                                           search_content="job")
    job_id = job_list_info["jobs"][0].job_id
    estimator = Estimator(session, job_id=job_id)
    job_version_info = estimator.get_job_version_info()
    version_id = job_version_info["versions"][0]["version_id"]
    version_status = job_version_info["versions"][0]["status"]
    if version_status == 8:
        estimator = Estimator(session, job_id=job_id, version_id=version_id)
        estimator.stop_job_version()
        print("{} stopped".format(time.time()))
        time.sleep(60)
    estimator = Estimator(
        modelarts_session=session,
        framework_type='PyTorch',  # AI引擎名称
        framework_version='PyTorch-1.0.0-python3.6',  # AI引擎版本
        code_dir='/ll-coco/codes/',  # 训练脚本目录
        boot_file='/ll-coco/codes/train.py',  # 训练启动脚本目录
        log_url='/ll-coco/dla/txtlogs/',  # 训练日志目录
        hyperparameters=[
            {"label": "cfg_path",
             "value": "s3://ll-coco/codes/configs/train_cfg.yaml"}
        ],
        output_path='/ll-coco/dla/',  # 训练输出目录
        train_instance_type='modelarts.vm.gpu.free',  # 训练环境规格
        train_instance_count=1)
    estimator.create_job_version(job_id=job_id,
                                 pre_version_id=version_id,
                                 inputs='/ll-coco/datasets/cityscapes/', wait=False,
                                 job_desc='train for net')
    print("{} created".format(time.time()))
    time.sleep(3600)


if __name__ == "__main__":
    # create_job()
    while True:
        create_loop()
