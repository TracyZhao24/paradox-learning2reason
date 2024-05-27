import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import json

# current experiment:
# experiment_directory = 'EXPERIMENTS/15_may_2024_1/'
experiment_directory = 'EXPERIMENTS/22_may_2024/'

# get experiment details
with open(experiment_directory + "experiment_details.json", "r") as file:
    details = json.load(file)
max_reasoning_depth = details["max_reasoning_depth"]
model_layers = details["model_layers"]

# file names
TRAIN_LOSS_LOG_FILE = 'train_loss_log.txt'
TEST_LOSS_LOG_FILE = 'test_loss_log.txt'
OTHER_DIST_LOSS_LOG_FILE = 'other_dist_loss_log.txt'
TRAIN_ACC_LOG_FILE = 'train_acc_log.txt'
TEST_ACC_LOG_FILE = 'test_acc_log.txt'
OTHER_DIST_ACC_LOG_FILE = 'other_dist_acc_log.txt'
PER_EPOCH_TRAIN_ACC_LOG_FILE = 'per_epoch_train_acc_log.txt'
PER_EPOCH_TEST_ACC_LOG_FILE = 'per_epoch_test_acc_log.txt'
PER_EPOCH_OTHER_DIST_ACC_LOG_FILE = 'per_epoch_other_dist_acc_log.txt'

# names of relevant files and corresponding names for each dataset
fnames = [[experiment_directory+TRAIN_LOSS_LOG_FILE,experiment_directory+TEST_LOSS_LOG_FILE,experiment_directory+OTHER_DIST_LOSS_LOG_FILE],
          [experiment_directory+TRAIN_ACC_LOG_FILE,experiment_directory+TEST_ACC_LOG_FILE,experiment_directory+OTHER_DIST_ACC_LOG_FILE]]
names = [['Train loss', 'Test loss', 'Other distribution loss'],
         ['Train accuracy', 'Test accuracy', 'Other distribution accuracy']]

fig, ax = plt.subplots(2, 3,sharex=True,sharey='row')#sharey='row' means each row will share the y axis within its row only; each row has its own y axis
fig.set_size_inches(12,8)

# for each dataset
for r in range(2):
    for c in range(3):
        fname = fnames[r][c]
        name = names[r][c]
        # parse the data
        accum = []
        losses_by_depth = [[] for i in range(max_reasoning_depth + 1)]
        with open(fname, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                entries = [i for i in line.split()] 
                # epoch number appears in the first entry, but we'll skip that for now
                accum.append(float(entries[1]))
                for depth in range(max_reasoning_depth+1):
                    losses_by_depth[depth].append(float(entries[2+depth]))
        # plot the data
        # if r == 0:
        #     curax = plt.subplot(2, 3, c+1, sharey=ax_row1)
        # else:
        #     curax = plt.subplot(2, 3, 3+c+1, sharey=ax_row2)
        curax = ax[r][c] #plt.subplot(2,3,3*r+c+1)
        curax.title.set_text(name)
        curax.set_xlabel("Batch number")
        if r == 0:
            curax.set_ylabel("Loss")
            
            for i in range(len(losses_by_depth)):
                # if len(losses_by_depth[i]) > 0:
                curax.plot(list(range(len(losses_by_depth[i]))),losses_by_depth[i], linewidth=.25, label ='Depth '+str(i))#+' loss')
            curax.plot(list(range(len(accum))), accum, label ='Batch loss', color='brown')

        else:
            curax.set_ylabel("Accuracy")
            
            for i in range(len(losses_by_depth)):
                # if len(losses_by_depth[i]) > 0:
                curax.plot(list(range(len(losses_by_depth[i]))),losses_by_depth[i], linewidth=.25, label ='Depth '+str(i))#+' accuracy')
            curax.plot(list(range(len(accum))), accum, label ='Batch accuracy', color='brown')
            # curax.set_ylim(top=1.01)
                
        curax.legend()

# last one is an accuracy and we want to set the max to 1.01
curax.set_ylim(top=1.01)

plt.suptitle("Max reasoning depth "+str(max_reasoning_depth)+". Model layers "+str(model_layers)+". Learning rate "+str(details["lr"])+". Batch size "+str(details["effective_batch_size"])+".")
plt.tight_layout()
plt.savefig(experiment_directory+'plot.png')