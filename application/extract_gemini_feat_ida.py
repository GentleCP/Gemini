#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------File Info-----------------------
Name: extract_gemini_feat_ida.py
Description: ida srcipt to extract gemini features, require python3
Author: GentleCP
Email: me@gentlecp.com
Create Date: 2022/9/28 
-----------------End-----------------------------
"""

import idaapi
import idautils
import idc
import time
import json
import networkx as nx
from tqdm import tqdm
from ida_pro import qexit
from collections import OrderedDict
from cptools import LogHandler
from pathlib import Path

OPTYPEOFFSET = 1000
IMM_MASK = 0xffffffff  # 立即数的掩码
# user defined op type
o_string = idc.o_imm + OPTYPEOFFSET
o_calls = OPTYPEOFFSET + 100
o_trans = OPTYPEOFFSET + 101  # Transfer instructions
o_arith = OPTYPEOFFSET + 102  # arithmetic instructions
o_all = OPTYPEOFFSET + 103  # all instruction

transfer_instructions_x86 = {'MOV', 'PUSH', 'POP', 'XCHG', 'IN', 'OUT', 'XLAT', 'LEA', 'LDS', 'LES', 'LAHF', 'SAHF',
                             'PUSHF', 'POPF'}
# https://cdrdv2.intel.com/v1/dl/getContent/671200
arithmetic_instructions_x86 = {'ADD', "ADC", "ADCX", "ADOX", "SBB", 'SUB', 'MUL', 'DIV', 'INC', 'DEC', 'IMUL', 'IDIV',
                               'CMP', "NEG",
                               "DAA", "DAS", "AAA", "AAS", "AAM", "AAD"}

# https://azeria-labs.com/assembly-basics-cheatsheet/
# https://www.ic.unicamp.br/~ranido/mc404/docs/ARMv7-cheat-sheet.pdf
transfer_instructions_arm = {"B", "BAL", "BNE", "BEQ", "BPL", "BMI", "BCC", "BLO", "BCS", "BHS", "BVC", "BVS", "BGT",
                             "BGE", "BLT", "BLE", "BHI", "BLS"}
arithmetic_instructions_arm = {"add", "adc", "qadd", "sub", "sbc", "rsb", "qsub", "mul", "mla", "mls", "umull", "umlal",
                               "smull",
                               "smlal", "udiv", "sdiv", "cmp", "cmn", "tst"}

# reference. https://uweb.engr.arizona.edu/~ece369/Resources/spim/MIPSReference.pdf
transfer_instructions_mips = {"beqz", "beq", "bne", "bgez", "b", "bnez", "bgtz", "bltz", "blez", "bgt", "bge", "blt",
                              "ble", "bgtu", "bgeu", "bltu", "bleu"}
arithmetic_instructions_mips = {"add", "addu", "addi", "addiu", "and", "andi", "div", "divu", "mult", "multu",
                                "slt", "sltu", "slti", "sltiu"}

# http://www.tentech.ca/downloads/other/PPC_Quick_Ref_Card-Rev1_Oct12_2010.pdf
# http://class.ece.iastate.edu/arun/CprE281_F05/lab/labw10a/Labw10a_Files/PowerPC%20Assembly%20Quick%20Reference.htm
transfer_instructions_ppc = {"b", "blt", "beq", "bge", "bgt", "blr", "bne"}
arithmetic_instructions_ppc = {"add", "addi", "addme", "addze", "neg", "subf", "subfic", "subfme", "subze", "mulhw",
                               "mulli",
                               "mullw", "divw", "cmp", "cmpi", "cmpl", "cmpli"}

transfer_instructions = ['MOV', 'PUSH', 'POP', 'XCHG', 'IN', 'OUT', 'XLAT', 'LEA', 'LDS', 'LES', 'LAHF', 'SAHF',
                         'PUSHF', 'POPF']
arithmetic_instructions = ['ADD', 'SUB', 'MUL', 'DIV', 'XOR', 'INC', 'DEC', 'IMUL', 'IDIV', 'OR', 'NOT', 'SLL', 'SRL']

ymd = time.strftime("%Y-%m-%d", time.localtime())
logger = LogHandler('Gemini')

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def wait_for_analysis_to_finish():
    """
    等待ida将二进制文件分析完毕再执行其他操作
    :return:
    """
    idaapi.auto_wait()


class FeatExtractor(object):

    def __init__(self):
        self._func_t = None

    def get_strings_of_block(self, block_start_ea, block_end_ea):
        return self.get_op_value_of_block(block_start_ea, block_end_ea, my_op_type=o_string)

    def get_trans_of_block(self, block_start_ea, block_end_ea):
        return self.get_op_value_of_block(block_start_ea, block_end_ea, o_trans)

    def get_all_ins_of_block(self, block_start_ea, block_end_ea):
        return self.get_op_value_of_block(block_start_ea, block_end_ea, o_all)

    # get operand value in one block
    def get_op_value_of_block(self, block_start_ea, block_end_ea, my_op_type):
        operands = []

        it_code = idaapi.func_item_iterator_t(self._func_t, block_start_ea)
        ea = it_code.current()
        while ea < block_end_ea:
            operands += self.get_op_value(ea, my_op_type)
            # see if arrive end of the blocks
            if not it_code.next_code():
                break
            ea = it_code.current()

        return operands

    def get_arith_of_block(self, block_start_ea, block_end_ea):
        return self.get_op_value_of_block(block_start_ea, block_end_ea, o_arith)

    # get immediate num in blocks
    def get_num_constants_of_block(self, block_start_ea, block_end_ea):
        return self.get_op_value_of_block(block_start_ea, block_end_ea, my_op_type=idc.o_imm)

    def get_callee_of_block(self, block_start_ea, block_end_ea):
        return self.get_op_value_of_block(block_start_ea, block_end_ea, my_op_type=o_calls)

    @staticmethod
    def get_op_value(ea, my_op_type=idc.o_void):
        """
        check value in ea is satisfy my_op_type or not, if True, return the values
        :param ea:
        :param my_op_type:
        :return:
        """
        op_values = []

        if my_op_type == o_trans:
            inst = idc.GetDisasm(ea).split(' ')[0].upper()
            if inst in transfer_instructions:
                op_values.append(inst)
            return op_values

        elif my_op_type == o_arith:
            inst = idc.GetDisasm(ea).split(' ')[0].upper()
            if inst in arithmetic_instructions:
                op_values.append(inst)
            return op_values
        elif my_op_type == o_all:
            inst = idc.GetDisasm(ea).split(' ')[0].upper()
            op_values.append(inst)
            return op_values
        op = 0

        op_type = idc.get_operand_type(ea, op)
        while op_type != idc.o_void:
            if my_op_type == o_calls:
                operands = idc.GetDisasm(ea).split(' ')
                if operands[0].upper() in ["CALL", "BL", "JAL", "B"]:
                    op_values.append(operands[-1])
                break

            if op_type == my_op_type % OPTYPEOFFSET:
                ov = idc.get_operand_value(ea, op)
                ov &= 0xffffffff  # 强制转化成32位
                if my_op_type == idc.o_imm:
                    if ov != 0:
                        op_values.append(hex(ov))
                elif my_op_type == o_string:
                    if not idc.get_segm_name(ov) == '.rodata':
                        addrx = list(idautils.DataRefsFrom(ov))
                        if len(addrx) == 0:
                            op += 1
                            op_type = idc.get_operand_type(ea, op)
                            continue
                        ov = addrx[0]
                    op_values.append(idc.GetString(ov))
            op += 1
            op_type = idc.get_operand_type(ea, op)
        return op_values

    @staticmethod
    def get_off_spring(graph):
        visit = set()

        def dfs(block_id):
            if block_id in visit:
                return 0
            visit.add(block_id)
            off_spring = 0
            for succ_node in graph.successors(block_id):
                if succ_node not in visit:
                    off_spring += dfs(succ_node) + 1
            return off_spring

        block_id2off_spring = {}

        for block in graph.nodes():
            block_id2off_spring[block] = dfs(block)

        return block_id2off_spring

    @staticmethod
    def get_blocks_and_succs(func_t):
        flowchart = idaapi.FlowChart(func_t)

        blocks = []
        succs = []
        for block in flowchart:
            # block id from 0 -> flowchart.size
            cfg_i = []
            blocks.append((block.id, block.start_ea, block.end_ea))
            for successor in block.succs():
                cfg_i.append(successor.id)
            succs.append(cfg_i)
        return blocks, succs

    def get_block_features(self, block, block_id2offspring):
        """
        获取单个block的所有feature
        :param block: block_id, block_start_ea, block_end_ea
        :param block_id2offspring:
        :return: 7 features [string, num_constant, trans, callee, ins_all, arith, offspring]
        """
        return (
            len(self.get_strings_of_block(block[1], block[2])),  #
            len(self.get_num_constants_of_block(block[1], block[2])),  #
            len(self.get_trans_of_block(block[1], block[2])),  #
            len(self.get_callee_of_block(block[1], block[2])),  #
            len(self.get_all_ins_of_block(block[1], block[2])),   #
            len(self.get_arith_of_block(block[1], block[2])),   #
            block_id2offspring.get(block[0], 0),
        )

    def gen_succs_features(self, func_t):
        """
        给定一个函数体，提取其Gemini相关特征
        :param func_t:
        :return: {'func_name': '', 'n_num': 0, 'succs': [[1,2], ...], 'features': [[1,2,3,4,5,6,7], ...]}
        """

        self._func_t = func_t
        blocks, succs = self.get_blocks_and_succs(func_t)
        G = nx.DiGraph()
        for start_node, end_nodes in enumerate(succs):
            G.add_node(start_node)
            for node in end_nodes:
                G.add_node(node)
                G.add_edge(start_node, node)
        block_id2offspring = self.get_off_spring(G)
        features = []
        for block in blocks:
            features.append(self.get_block_features(block, block_id2offspring))
        while len(succs) > len(features):
            features.append((0, 0, 0, 0, 0, 0, 0))
        return {
            'func_name': idaapi.get_func_name(func_t.start_ea),
            'n_num': len(succs),
            'succs': succs,
            'features': features,
        }


def main():
    bin_name = idaapi.get_root_filename()
    if len(idc.ARGV) == 2:
        feat_path = Path(idc.ARGV[1])

    else:
        feat_path = Path('{}_Gemini_features.json'.format(bin_name))

    select_func_name_path = feat_path.parent.joinpath(f"{bin_name}_select_func_names.json")
    if select_func_name_path.exists():
        select_func_names = read_json(select_func_name_path)
    else:
        select_func_names = None

    extractor = FeatExtractor()
    with open(feat_path, 'w') as f:
        for i in tqdm(range(idaapi.get_func_qty())):
            func = idaapi.getn_func(i)
            func_name = idaapi.get_func_name(func.start_ea)
            if select_func_names and func_name not in select_func_names:
                continue
            seg_name = idc.get_segm_name(func.start_ea)
            if seg_name[1:3] not in ["OA", "OM", "te"]:
                continue

            data = extractor.gen_succs_features(func_t=func)
            f.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    # test_caller()
    try:
        wait_for_analysis_to_finish()
        main()
    except Exception as e:
        import traceback

        logger.error(traceback.format_exc())
    qexit(0)
