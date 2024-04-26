import matplotlib.pyplot as plt 
  
accum = [] 
depth_0 = []
depth_1 = []
depth_2 = [] 
depth_3 = [] 
with open('./logs/LP/loss_log_3.txt', 'r') as f:
    # next(f)
    lines = f.readlines()
    slice_lines = lines[1950:2050]
    # slice_lines = [next(f) for _ in range(300)]
    for line in slice_lines:
        lines = [i for i in line.split()] 
        # epoch.append(lines[0]) 
        accum.append(float(lines[1]))
        depth_0.append(float(lines[2]))
        depth_1.append(float(lines[3]))
        depth_2.append(float(lines[4]))
        depth_3.append(float(lines[5]))
      

plt.title("LP Model Loss by Depth") 
plt.xlabel("Batches") 
plt.ylabel("Loss") 

plt.plot(accum, label ='accumulated batch loss')
plt.plot(depth_0, label ='depth 0 batch loss')
plt.plot(depth_1, label ='depth 1 batch loss')
plt.plot(depth_2, label ='depth 2 batch loss')
plt.plot(depth_3, label ='depth 3 batch loss')
plt.legend()

# plt.yticks(y) 
# plt.plot(x, y, marker = 'o', c = 'g') 
  
plt.savefig('./plots/lp_slice_loss.png')