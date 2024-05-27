import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure

# experiment details
date_str = 'may_13'
max_reasoning_depth = 6

# names of relevant files and corresponding names for each dataset
fnames = ['./logs/LP/train_loss_log_'+date_str+'.txt','./logs/LP/test_loss_log_'+date_str+'.txt','./logs/LP/other_dist_loss_log_'+date_str+'.txt']
names = ['Train loss', 'Test loss', 'Other distribution loss']

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(14, 5)
k = 0
# for each dataset
for k,(fname,name) in enumerate(zip(fnames, names)):
    # parse the data
    accum = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]
        NUM_DEPTHS = 10
        losses_by_depth = [[] for i in range(NUM_DEPTHS)]
        for line in lines:
            entries = [i for i in line.split()] 
            # epoch.append(lines[0])
            accum.append(float(entries[1]))
            for depth in range(len(entries)):
                if depth==0:
                    continue
                losses_by_depth[depth].append(float(entries[depth]))
    # plot the data
    curax = ax[k]
    curax.title.set_text(name)
    curax.set_xlabel("Batches")
    curax.set_ylabel("Loss")

    curax.plot(list(range(len(accum))),accum, label ='accumulated batch loss')
    for i in range(len(losses_by_depth)):
        if len(losses_by_depth[i]) > 0:
            curax.plot(list(range(len(losses_by_depth[i]))),losses_by_depth[i], label ='Depth '+str(i)+' loss')
    curax.legend()

plt.suptitle("Max reasoning depth "+str(max_reasoning_depth)+". Model layers "+str(max_reasoning_depth+2)+".")
plt.savefig('./plots/'+date_str+'_losses.png')