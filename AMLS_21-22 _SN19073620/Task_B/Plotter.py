"""This code is used to plot the experimental results"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% dynamic learning rate searching

a1 = pd.read_csv('reg=0.csv',header=None)
a2 = pd.read_csv('reg=1e-4.csv',header=None)
a3 = pd.read_csv('dropout_only.csv',header=None)
lr = [1e-5]
for i in range(45):
    lr.append(lr[i] + lr[0] * pow(10, np.floor((i) / 9)))

plt.figure(figsize=(10, 5.2))
# plotting loss vs lr
plt.subplot(1, 2, 1)
plt.plot(lr,a1,label='original ')
plt.plot(lr,a2,label='regularization ')
plt.plot(lr,a3,label='dropout')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('learning rate (log scale)')
plt.ylabel('training loss (log scale)')
plt.legend(loc=2)



# plotting epochs vs lr
plt.subplot(1, 2, 2)
plt.plot(np.linspace(1,46,46),lr)
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('learning rate (log scale)')
# plt.savefig('lr.pdf',dpi=600)
plt.show()

#%% acc among each 10 models with different hyperparameters
a1 = pd.read_csv('no_scores.csv',header=None)
a2 = pd.read_csv('reg_scores.csv',header=None)
a3 = pd.read_csv('dropout_scores.csv',header=None)
a4 = pd.read_csv('all_scores_1e-1.csv',header=None)
a5 = pd.read_csv('all_scores_1e-2.csv',header=None)
a6 = pd.read_csv('all_scores_1e-3.csv',header=None)
a7 = pd.read_csv('all_scores_1e-4.csv',header=None)

m = [x for x in range(1,11)]


plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.plot(m,a1,label='original ')
plt.plot(m,a2,label='original ')
plt.plot(m,a3,label='original ')
# plt.plot(m,a4,label='original ')
# plt.plot(m,a5,label='original ')
# plt.plot(m,a6,label='original ')
# plt.plot(m,a7,label='original ')
plt.grid()
plt.show()
#%% avg acc among models with different hyperparameters

plt.figure(figsize=(7, 4.8))
plt.subplot(1, 2, 1)
a = np.reshape(np.array([a1,a2,a3,a4,a5,a6,a7]),[7,10])
a = np.mean(a,1)
# num_list = a
name_list = ['original','reg','dropout','dropout\n+reg(1e-1)'
             ,'dropout\n+reg(1e-2)','dropout\n+reg(1e-3)','dropout\n +reg(1e-4)']

plt.barh(['original','reg','dropout','dropout\n+reg(1e-1)'
             ,'dropout\n+reg(1e-2)','dropout\n+reg(1e-3)','dropout\n +reg(1e-4)'],a,ec='r',ls='--')

for x, y in zip(name_list, a):
    plt.text(y+1e-2, x, round(y,3), ha='center', va='bottom')


plt.xlim(0.90,0.98)
plt.xlabel('avg val_acc')


plt.subplot(1, 2, 2)

plt.subplot(1, 2, 2).yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
total_width, n = 0.8, 2
width = total_width / n
m = np.array([x for x in range(1,11)])
m = m - (total_width - width) / 2
plt.xticks([1,2,3,4,5,6,7,8,9,10])

plt.bar(m, np.array(a3)[:,0],width=width , label='dropout')
plt.bar(m + width, np.array(a6)[:,0],width=width, label='dropout+reg(1e-3)')
plt.ylim(0.94,1)
plt.xlabel('fold No. ')
plt.ylabel('val_acc ')
plt.legend(loc=2)
# plt.savefig('eva-10cv.pdf',dpi=600)
plt.show()

#%% acc & loss of model A and model B

a1 = np.array(pd.read_csv('dropout_acc.csv',header=None))
a2 = np.array(pd.read_csv('dropout_loss.csv',header=None))

a3 = np.array(pd.read_csv('all_acc_1e-3.csv',header=None))
a4 = np.array(pd.read_csv('all_loss_1e-3.csv',header=None))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(a1[0,:], label='accuracy (A)')
plt.plot(a1[1,:], label = 'val_accuracy(A)')
plt.plot(a3[0,:], label='accuracy(B)')
plt.plot(a3[1,:], label = 'val_accuracy(B)')
plt.xlabel('epoch')
plt.ylabel('accuracy')
# plt.ylim([0.9, 1])
# plt.grid()
plt.legend(loc=4)

plt.subplot(1, 2, 2)
plt.plot(a2[0,:], label='loss (A)')
plt.plot(a2[1,:], label = 'val_loss(A)')
plt.plot(a4[0,:], label='loss(B)')
plt.plot(a4[1,:], label = 'val_loss(B)')
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.ylim([0.9, 1])
# plt.grid()
plt.legend(loc=1)

# plt.savefig('accu_loss.pdf',dpi=600)
plt.show()


#%%










































