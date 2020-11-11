
f_label = open("label.txt",'r')
f_translate = open("translate.txt",'r')
f_flag = open("flag.txt",'r')

for line_label, line_trans in zip(f_label,f_translate):
    line_label=line_label.strip().split()
    line_trans=line_trans.strip().split()
    label_length = len(line_label)
    flag_line = []
    for i, word in enumerate(line_trans):
        piece = line_label[max(0,i-2):min(i+2,label_length)]
        if word in piece:
            flag_line.append(1)
        else:
            flag_line.append(-1)
    print(flag_line,file=f_flag)