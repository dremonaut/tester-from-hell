import os
os.environ['KERAS_BACKEND'] = "theano"


import pickle
from example.smart_vacuum_system.svs_gui import SVSVisualizer

# Load failure dump
with open('experiment/logs/dfp_failure_data_test.pkl', 'rb') as input:
    failure_list = pickle.load(input)

print(len(failure_list))

# Visualize
#print(failure_list)
record = failure_list[2][1].record
gui = SVSVisualizer()
for state in record:
    gui.put_state(state)
gui.run()

#for failure in failure_list[0]:
#    gui.put_state(failure.record[-1])
#    gui.run()
