#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------File Info-----------------------
Name: extract_gemini_feat.py
Description: interface for generate gemini features
require python3
Author: GentleCP
Email: me@gentlecp.com
Create Date: 2022/9/30 
-----------------End-----------------------------
"""
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from tqdm import tqdm
import subprocess

IDA_PATH = '/home/cp/Application/idapro-7.5/idat64'


def execute_cmd(cmd, timeout=900):
    """
    execute system command
    :param cmd:
    :param f: 用于指定输出到文件显示，方便后台追踪长时间运行的程序
    :param timeout:
    :return:
    """
    try:
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)

    except subprocess.TimeoutExpired as e:
        return {
            'errcode': 401,
            'errmsg': 'timeout'
        }
    return {
        'errcode': p.returncode,
        'errmsg': p.stdout.decode()
    }


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def extract_by_bin_path(bin_path):
    """
    提取单个binary_path下的gemini features
    :param bin_path:
    :param path2func_names_path:
    :return:
    """
    bin_path = Path(bin_path)
    feature_path = bin_path.parent.joinpath(f'{bin_path.name}_Gemini_features.json')
    # if feature_path.exists():
    #     return {
    #         'errcode': 0,
    #         'bin_path': str(bin_path),
    #         'errmsg': 'exist'
    #     }
    cmd = f'TVHEADLESS=1 {IDA_PATH} -Llog/gemini_features_ida.log -c -A -S"./extract_gemini_feat_ida.py {feature_path}" {bin_path}'
    exe_res = execute_cmd(cmd, timeout=3600)
    exe_res['bin_path'] = str(bin_path)
    return exe_res


def load_paths():
    path2func_names_path = Path('../data/filter_path2func_names.json')
    assert path2func_names_path.exists(), f"file not found:{path2func_names_path}"
    path2func_names = read_json(path2func_names_path)
    return [path for path in path2func_names.keys() if path.split('/')[0] in ['X86', 'ARM', 'MIPS']]


def main():
    base_path = Path('/home/cp/dataset/buildroot-elf-5arch')
    ext_results = defaultdict(list)
    bar = tqdm(load_paths())
    for path in bar:
        res = extract_by_bin_path(base_path.joinpath(path))
        if res['errcode'] == 0:
            ext_results['success'].append(res)
        else:
            ext_results['fail'].append(res)
        bar.set_description(f'success: {len(ext_results["success"])}, fail:{len(ext_results["fail"])}')
    write_json(ext_results, 'extract_results.json')


if __name__ == '__main__':
    main()
