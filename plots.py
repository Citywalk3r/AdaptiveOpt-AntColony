import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# plt.style.use(['seaborn-white', 'seaborn-paper'])
matplotlib.rc("font", family="monospace")
# import seaborn as sns

column_names = ['iterations', 'h', 776, 12, 234, 9238, 123556, 59933, 98232, 85732, 5432, 12291]

df = pd.read_excel("../tabu_10_seeds_t11_random_restart_best_so_far.xlsx")
problem = "QAP"

fig = plt.figure(figsize=(10, 5))

# Effect of ps
grouped = pd.melt(df, id_vars=["h"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["h"] == 9]
plt.subplot(1,3,1)
plt.title('h = 9')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["h"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["h"] == 10]
plt.subplot(1,3,2)
plt.title('h = 10')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["h"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["h"] == 11]
plt.subplot(1,3,3)
plt.title('h = 11')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

plt.suptitle("Impact of tabu list size (h) on TS for the " + problem)
plt.show()


# Effect of size change frequency
grouped = pd.melt(df, id_vars=["restart f"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["restart f"] == 10]
plt.subplot(1,3,1)
plt.title('random restart every = 10 iterations')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["restart f"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["restart f"] == 50]
plt.subplot(1,3,2)
plt.title('random restart every = 50 iterations')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["restart f"],value_vars=["776", "12", "234", "9238", "123556", "59933", "98232", "85732", "5432", "12291"])
grouped = grouped[grouped["restart f"] == 100]
plt.subplot(1,3,3)
plt.title('random restart every = 100 iterations')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

plt.suptitle("Impact of random restart frequency on TS for the " + problem)
plt.show()


# Effect of gmax
fig = plt.figure(figsize=(10, 5))

grouped = pd.melt(df, id_vars=["gmax"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["gmax"] == 5]
print(grouped)
plt.subplot(2,3,1)
plt.title('gmax = 5')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["gmax"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["gmax"] == 20]
plt.subplot(2,3,2)
plt.title('gmax = 20')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["gmax"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["gmax"] == 50]
plt.subplot(2,3,3)
plt.title('gmax = 50')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["gmax"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["gmax"] == 100]
plt.subplot(2,3,4)
plt.title('gmax = 100')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["gmax"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["gmax"] == 200]
plt.subplot(2,3,5)
plt.title('gmax = 200')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["gmax"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["gmax"] == 500]
plt.subplot(2,3,6)
plt.title('gmax = 500')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")


plt.suptitle("Impact of g_max on DE for the " + problem)
plt.show()

# Effect of F
fig = plt.figure(figsize=(10, 5))

grouped = pd.melt(df, id_vars=["F"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["F"] == 0.1]
print(grouped)
plt.subplot(2,3,1)
plt.title('F = 0.1')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["F"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["F"] == 0.2]
plt.subplot(2,3,2)
plt.title('F = 0.2')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["F"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["F"] == 0.4]
plt.subplot(2,3,3)
plt.title('F = 0.4')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["F"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["F"] == 0.6]
plt.subplot(2,3,4)
plt.title('F = 0.6')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["F"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["F"] == 0.7]
plt.subplot(2,3,5)
plt.title('F = 0.7')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["F"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["F"] == 0.9]
plt.subplot(2,3,6)
plt.title('F = 0.9')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")


plt.suptitle("Impact of F on DE for the " + problem)
plt.show()

# Effect of F
fig = plt.figure(figsize=(10, 5))

grouped = pd.melt(df, id_vars=["CR"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["CR"] == 0.2]
print(grouped)
plt.subplot(2,3,1)
plt.title('CR = 0.2')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["CR"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["CR"] == 0.4]
plt.subplot(2,3,2)
plt.title('CR = 0.4')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["CR"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["CR"] == 0.5]
plt.subplot(2,3,3)
plt.title('CR = 0.5')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["CR"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["CR"] == 0.6]
plt.subplot(2,3,4)
plt.title('CR = 0.6')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["CR"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["CR"] == 0.8]
plt.subplot(2,3,5)
plt.title('CR = 0.8')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["CR"],value_vars=["776", "1", "234", "9238", "123556"])
grouped = grouped[grouped["CR"] == 0.9]
plt.subplot(2,3,6)
plt.title('CR = 0.9')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")


plt.suptitle("Impact of CR on DE for the " + problem)
plt.show()