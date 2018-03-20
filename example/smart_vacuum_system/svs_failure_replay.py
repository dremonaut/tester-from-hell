import pickle
from example.smart_vacuum_system.svs_gui import SVSVisualizer

# Load failure dump
with open('failure_data.pkl', 'rb') as input:
    failure_list = pickle.load(input)

print(len(failure_list))

# Visualize
record = failure_list[78].record
gui = SVSVisualizer()
for state in record:
    gui.put_state(state)
gui.run()
