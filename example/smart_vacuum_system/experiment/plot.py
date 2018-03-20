import matplotlib.pyplot as plt
import json
import numpy as np

steps = np.linspace(0, 100000, 100)

# Load kpi s

with open('standard_tester.json') as st_data_file:
    standard_tester_data = json.load(st_data_file)
with open('independent_tester.json') as it_data_file:
    independent_tester_data = json.load(it_data_file)
with open('random_tester.json') as rt_data_file:
    random_tester_data = json.load(rt_data_file)

plt.plot(steps, standard_tester_data['avgs'])
plt.plot(steps, independent_tester_data['avgs'])
plt.plot(steps, random_tester_data['avgs'])

plt.ylabel('avg no. revealed faults',fontsize=12)
plt.xlabel('test steps', fontsize=12)

plt.legend(['heuristic', 'independent', 'random'], loc='upper left')

plt.show()