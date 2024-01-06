def Get_Max_From_Matrix2(x):
    max = -100.0
    for i in x:
        for j in i:
            if float(j) > max:
                max = float(j)
    return max


def Get_Min_From_Matrix2(x):
    min = 100.0
    for i in x:
        for j in i:
            if float(j) < min:
                min = float(j)
    return min


def Get_Average_From_Matrix2(x):
    print(Get_Max_From_Matrix2(x))
    print(Get_Min_From_Matrix2(x))
    return (Get_Max_From_Matrix2(x) + Get_Min_From_Matrix2(x)) / 2
