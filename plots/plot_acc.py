import matplotlib.pyplot as plt 
  
epoch = [] 
train = []
validate = []
test = [] 
with open('./logs/LP/acc_log_3.txt', 'r') as f:
    next(f)
    for line in f:
        lines = [i for i in line.split()] 
        epoch.append(lines[0]) 
        train.append(float(lines[1]))
        validate.append(float(lines[2]))
        test.append(float(lines[3]))
      
plt.title("LP Model Accuracy") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 

plt.plot(epoch, train, '-.', label ='train accuracy', color='b')
plt.plot(epoch, validate, '.', label ='validate accuracy', color='g')
plt.plot(epoch, test, '--', label ='test accuracy', color='r')
plt.legend()

# plt.yticks(y) 
# plt.plot(x, y, marker = 'o', c = 'g') 
  
plt.savefig('lp_accuracy.png')