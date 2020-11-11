ref = []
out = []
split_ref=dict()
split_out=dict()
model_name='base'
# with open('/home/user_data/wengrx/WMT16_ENDE/newstest2016.tok.en','r') as f:
#     for line in f:
#         ref.append(line.strip().split())
# print(ref[2])
# with open('../result/deen_wpalat/deen_wpalat.newstest2017.txt.0','r') as f:
with open('/home/wengrx/Projects/NJUNMT-pytorch-master/result_zhen/transformer.mt03.txt.0','r') as f:
    for line in f:
        out.append(line.strip().split())
fl =  open('mt03.base.deen.out.left','a')
fr =  open('mt03.base.deen.out.right','a')

for sentence in out:
    midlen=int(len(sentence)/2)
    print(' '.join(sentence[:midlen]),file=fl)
    print(' '.join(sentence[midlen:]),file=fr)

fl.close()
fr.close()
# print(out[2])
# for i, sentence in enumerate(ref):
#     if len(sentence) < 15:
#         with open('nst2016.ref.011','a') as f:
#             print(' '.join(sentence),file=f)
#         with open('nst2016.out.'+model_name+'.011','a') as f:
#             print(' '.join(out[i]),file=f)
#     elif 15<= len(sentence) < 30:
#         with open('nst2016.ref.021','a') as f:
#             print(' '.join(sentence),file=f)
#         with open('nst2016.out.'+model_name+'.021','a') as f:
#             print(' '.join(out[i]),file=f)
#     elif 30 <= len(sentence) < 45:
#         with open('nst2016.ref.031', 'a') as f:
#             print(' '.join(sentence), file=f)
#         with open('nst2016.out.'+model_name+'.031', 'a') as f:
#             print(' '.join(out[i]), file=f)
#     else:
#         with open('nst2016.ref.041', 'a') as f:
#             print(' '.join(sentence), file=f)
#         with open('nst2016.out.'+model_name+'.041', 'a') as f:
#             print(' '.join(out[i]), file=f)