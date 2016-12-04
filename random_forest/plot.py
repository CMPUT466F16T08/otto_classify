import matplotlib.pyplot as plt

n_estimators_list = [1, 3, 5, 8, 10, 20, 50, 80, 100, 150]
time_list = [3.10840201378, 5.73491287231, 7.72790789604, 9.67133188248, 10.9803099632, 18.4337930679, 42.0222539902, 65.9724440575, 79.2345659733, 124.668261051]
logloss_list = [0.983584942104, 0.72370282791, 0.639800705049, 0.587190351508, 0.567693122197, 0.52659471234, 0.497001862431, 0.494408305501, 0.489758294333, 0.490379771795]

# http://stackoverflow.com/questions/15082682/matplotlib-diagrams-with-2-y-axis
ax = plt.gca()
#ax2 = ax.twinx()

plt.axis('normal')
ax.plot(n_estimators_list, logloss_list, 'r',linewidth=1.5)
ax.set_ylabel("log loss",fontsize=14,color='red')
#ax2.plot(n_estimators_list, time_list, 'b',linewidth=1.5)
#ax2.set_ylabel("time(s)",fontsize=14,color='blue')

plt.title("RF", fontsize=20,color='black')
ax.set_xlabel('n_estimators', fontsize=14, color='black')
plt.show()