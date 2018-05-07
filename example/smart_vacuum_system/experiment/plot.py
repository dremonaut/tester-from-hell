import matplotlib.pyplot as plt
import json
import numpy as np
from _operator import add, sub

steps = np.linspace(0, 10000, 100)
alpha = 0.1


def sub_max_0(x, y):
    return max(0.0, x - y)
v_func = np.vectorize(sub_max_0)


def plot_failure_detection():
    # Load kpi
    with open('logs/q_learning_tester_step_1.json') as st_data_file:
        q_learning_tester_data = json.load(st_data_file)
        q_avg = np.array(q_learning_tester_data['avgs'])
        q_stddev = np.array(q_learning_tester_data['stddev'])
    with open('logs/dfp_tester_step_1.json') as it_data_file:
        dfp_tester_data = json.load(it_data_file)
        dfp_avg = np.array(dfp_tester_data['avgs'])
        dfp_stddev = np.array(dfp_tester_data['stddev'])
    with open('logs/random_tester_step_1.json') as rt_data_file:
        random_tester_data = json.load(rt_data_file)
        random_avg = np.array(random_tester_data['avgs'])
        random_stddev = np.array(random_tester_data['stddev'])

    fig, ax = plt.subplots(1)
    ax.plot(steps, dfp_avg, lw=2, label='DFP Tester', color='blue')
    ax.plot(steps, q_avg, lw=2, label='DQL Tester', color='red')
    ax.plot(steps, random_avg, lw=2, label='Random Tester', color='green')
    ax.fill_between(steps, dfp_avg + dfp_stddev,
                    v_func(dfp_avg, dfp_stddev), facecolor='blue', alpha=alpha)
    ax.fill_between(steps, q_avg + q_stddev, v_func(q_avg, q_stddev),
                    facecolor='red', alpha=alpha)
    ax.fill_between(steps, random_avg + random_stddev,
                    v_func(random_avg, random_stddev), facecolor='green', alpha=alpha)
    #ax.set_title('random walkers empirical $\mu$ and $\pm \sigma$ interval')
    ax.legend(loc='upper left')
    ax.set_xlabel('test steps')
    ax.set_ylabel('number of revealed faults')
    ax.grid()
    ax.plot()

    plt.show()

    fig.tight_layout()
    fig.savefig('C:/Users/reichsan/Desktop/rbt_sas_plots/failure_detection.png', bbox_inches='tight')


def plot_interstep():
    # Load kpi
    with open('logs/dfp_tester_step_2.json') as it_data_file:
        dfp_tester_data = json.load(it_data_file)
        dfp_avg = np.array(dfp_tester_data['avgs'])
        dfp_stddev = np.array(dfp_tester_data['stddev'])
    with open('logs/dfp_tester_step_2_free.json') as rt_data_file:
        dfp_tester_free_data = json.load(rt_data_file)
        dfp_free_avg = np.array(dfp_tester_free_data['avgs'])
        dfp_free_stddev = np.array(dfp_tester_free_data['stddev'])

    fig, ax = plt.subplots(1)
    ax.plot(steps, dfp_avg, lw=2, label='Pre-Trained DFP Tester', color='blue')
    ax.plot(steps, dfp_free_avg, lw=2, label='Untrained DFP Tester', color='orange')
    ax.fill_between(steps, dfp_avg + dfp_stddev,
                    v_func(dfp_avg, dfp_stddev), facecolor='blue', alpha=alpha)
    ax.fill_between(steps, dfp_free_avg + dfp_free_stddev,
                    v_func(dfp_free_avg, dfp_free_stddev), facecolor='orange', alpha=alpha)
    # ax.set_title('random walkers empirical $\mu$ and $\pm \sigma$ interval')
    ax.legend(loc='upper left')
    ax.set_xlabel('test steps')
    ax.set_ylabel('number of revealed faults')
    ax.grid()

    plt.show()

    fig.tight_layout()
    fig.savefig('C:/Users/reichsan/Desktop/rbt_sas_plots/interstep.png', bbox_inches='tight')


def plot_goal_generalization():
    # Load kpi

    with open('logs/dfp_tester_step_generalization.json') as it_data_file:
        dfp_tester_data = json.load(it_data_file)
        dfp_avg = np.array(dfp_tester_data['avgs'])
        dfp_stddev = np.array(dfp_tester_data['stddev'])
    with open('logs/q_learning_tester_step_generalization.json') as rt_data_file:
        dql_tester_data = json.load(rt_data_file)
        dql_avg = np.array(dql_tester_data['avgs'])
        dql_stddev = np.array(dql_tester_data['stddev'])

    fig, ax = plt.subplots(1)
    ax.plot(steps, dfp_avg, lw=2, label='DFP Tester', color='blue')
    ax.plot(steps, dql_avg, lw=2, label='DQL Tester', color='red')
    ax.fill_between(steps, dfp_avg + dfp_stddev,
                    v_func(dfp_avg, dfp_stddev), facecolor='blue', alpha=alpha)
    ax.fill_between(steps, dql_avg + dql_stddev,
                    v_func(dql_avg, dql_stddev), facecolor='red', alpha=alpha)
    # ax.set_title('random walkers empirical $\mu$ and $\pm \sigma$ interval')
    ax.legend(loc='upper left')
    ax.set_xlabel('test steps')
    ax.set_ylabel('number of revealed faults')
    ax.grid()

    plt.show()

    fig.tight_layout()
    fig.savefig('C:/Users/reichsan/Desktop/rbt_sas_plots/goal_generalization.png', bbox_inches='tight')


plot_failure_detection()
plot_goal_generalization()
plot_interstep()
