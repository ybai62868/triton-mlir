from enum import Enum


class Var1(Enum):
    VAL1 = 256
    VAL2 = 128
    VAL3 = 64
    VAL4 = 32
    VAL5 = 16


class Var2(Enum):
    VAL1 = 256
    VAL2 = 128
    VAL3 = 64
    VAL4 = 32
    VAL5 = 16


class Var3(Enum):
    VAL1 = 1024
    VAL2 = 512
    VAL3 = 256
    VAL4 = 128
    VAL5 = 64
    VAL6 = 32
    VAL7 = 16
    VAL8 = 8


class Var4(Enum):
    VAL1 = 32
    VAL2 = 16
    VAL3 = 8
    VAL4 = 4
    VAL5 = 2
    VAL6 = 1


class Var5(Enum):
    VAL1 = 32
    VAL2 = 16
    VAL3 = 8
    VAL4 = 4
    VAL5 = 2
    VAL6 = 1
    


class Var6(Enum):
    VAL1 = 32
    VAL2 = 16
    VAL3 = 8
    VAL4 = 4
    VAL5 = 2
    VAL6 = 1


class Var7(Enum):
    VAL1 = 16
    VAL2 = 8
    VAL3 = 1

class Var8(Enum):
    VAL1 = 8
    VAL2 = 1


def generate_table():
    # combinations = []
    table = {}
    table['BLOCK_SIZE_M'] = [var.value for var in Var1]
    table['BLOCK_SIZE_N'] = [var.value for var in Var2]
    table['BLOCK_SIZE_K'] = [var.value for var in Var3]
    table['num_warps'] = [var.value for var in Var4]
    table['WARPS_PER_TILE_0'] = [var.value for var in Var5]
    table['WARPS_PER_TILE_1'] = [var.value for var in Var6]
    table['SHAPE_PER_WARP_0'] = [var.value for var in Var7]
    table['SHAPE_PER_WARP_1'] = [var.value for var in Var8]
    return table


table = generate_table()
print(table)


def generate_combinations():
    combinations = []
    for var1 in Var1:
        for var2 in Var2:
            for var3 in Var3:
                for var4 in Var4:
                    for var5 in Var5:
                        for var6 in Var6:
                            for var7 in Var7:
                                for var8 in Var8:
                                    combination = (var1.value, var2.value, var3.value, var4.value, var5.value, var6.value, var7.value, var8.value)
                                    combinations.append(combination)
    return combinations


combinations = generate_combinations()
print(len(combinations))
