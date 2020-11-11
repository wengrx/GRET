import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heat_map(x, figure_no):
    plt.figure(figure_no)
    plt.pcolor(x)
    plt.colorbar()
    plt.show()


def generate(df,y):
    y=y[:df.shape[1]-1]
    print(df.shape[1])
    print(len(y))
    # print(df.shape)

    # dfData = df.corr()
    plt.subplots(figsize=(df.shape[1], df.shape[0]))  # 设置画面大小
    fig = plt.figure()
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=2)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    sns.heatmap(df, annot=False, vmax=0.55, square=False, cmap="Blues",cbar=True,xticklabels=False,yticklabels=False)
    plt.savefig('./heat.png')
    plt.show()


if __name__ == '__main__':
    plt.close('all')
    # x = np.random.normal(loc=0.5, scale=0.2, size=(9, 9))
    # test(x)
    x = np.load('gate.npy')
    trg=[]
    with open('trg.txt','r') as f:
        for line in f:
            trg.append(line.strip().split())

    # x= np.array([[0.2896, 0.2108, 0.1934, 0.1401, 0.0659, 0.1002],
    #              [0.1452, 0.1673, 0.2833, 0.2648, 0.0329, 0.1065],
    #              [0.0297, 0.1694, 0.3104, 0.2767, 0.0039, 0.2097],
    #              [0.0133, 0.1442, 0.3023, 0.2972, 0.0486, 0.1944],
    #              [0.0225, 0.1341, 0.3003, 0.2821, 0.0651, 0.1959],
    #              # [0.1546, 0.2007, 0.2563, 0.0027, 0.1415, 0.2443],
    #              [0.1196, 0.1607, 0.1957, 0.2329, 0.0758, 0.2153]])
    # x=x[::-1]
    # x=[i[::-1] for i in x]
    # x=x.transpose()
    # x = [i[::-1] for i in x]

    for num,i in enumerate(x):
        x=[]
        for j in i:
            x.append(list(j))
        y=np.array(x)
        y=y.sum(0)/8
        # print(y.shape)
        # print(max(y))
        # print(min(y))
        x=list(y)
        # print(type(i[0]))
        # print(i.reshape([8,len(i[0])]).shape)
        generate(np.matrix(x),trg[num])
        z=input()
    # plot_heat_map(x,2)

# import matplotlib.pyplot as plt
