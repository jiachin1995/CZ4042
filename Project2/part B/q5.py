import pylab as plt

test_acc = [
    0.5985714,
    0.85,
    0.72,
    0.7871429,
    0.6414286,
    0.9028571,
    0.71,
    0.89,
]

runtimes = [
    1106,
    183,
    439,
    141,
    1030,
    186,
    426,
    138,
]


labels = [
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q1&dropouts",
    "Q2&dropouts",
    "Q3&dropouts",
    "Q4&dropouts",
]

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(labels, test_acc, 'r-')
ax2.plot(labels, runtimes, 'b-')

ax1.set_xlabel('X data')
ax1.set_ylabel('test accuracy', color='red')
ax2.set_ylabel('Run Time (seconds)', color='b')




plt.savefig('figures/q5.png')
plt.show()

