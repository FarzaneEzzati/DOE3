"""
This is the code to visualize the day-ahead planning in the form of power exchanges hapenning.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl


def saveDAToCSV(SH):
    with open('results/dayAheadMGs.pkl', 'rb') as handle:
        mg_Ymax, mg_Lmax, mg_Z,  mg_Xes, mg_Xpv, mg_Xdg = pkl.load(handle)
    handle.close()
    with open('results/dayAheadCEMS.pkl', 'rb') as handle:
        V, R, K, _EA, _ER, _ES = pkl.load(handle)
    handle.close()

    print(f'Energy Access: {100*_EA:0.2f}%, Energy Resilience: {100*_ER:0.2f}%, Energy Sustainability: {100*_ES:0.2f}%.')

    bigTable = {'Time': range(SH)}
    for i in mg_Z.keys():
        bigTable[f'Z from {i}'] = mg_Z[i].values()
    for i in V.keys():
        bigTable[f'V to {i}'] = V[i].values()
    for i in R.keys():
        bigTable[f'R to {i}'] = R[i].values()
    for i in K.keys():
        bigTable[f'K {i}'] = K[i].values()
    for i in mg_Ymax.keys():
        bigTable[f'Max Shed {i}'] = mg_Ymax[i].values()
    pd.DataFrame(bigTable).to_csv('results/dayAheadPlan.csv', index=False)



def saveRTToCSV(SH):
    with open('results/realTimeMGs.pkl', 'rb') as handle:
        mg_Obj_rt, mg_Z_rt, mg_Y_rt = pkl.load(handle)
    handle.close()
    with open('results/realTimeCEMS.pkl', 'rb') as handle:
        cems_Obj_rt, cems_V_rt = pkl.load(handle)
    handle.close()

    bigTable = {'Time': range(SH)}
    for i in mg_Z_rt.keys():
        bigTable[f'Z from {i}'] = mg_Z_rt[i].values()
    for i in cems_V_rt.keys():
        bigTable[f'V to {i}'] = cems_V_rt[i].values()
    for i in mg_Y_rt.keys():
        bigTable[f'Load Shed {i}'] = mg_Y_rt[i].values()

    pd.DataFrame(bigTable).to_csv('results/realTimePlan.csv', index=False)


def plotDAvsRT(MG):
    DA_data = pd.read_csv(f'results/dayAheadPlan.csv', index_col='Time')
    RT_data = pd.read_csv(f'results/realTimePlan.csv', index_col='Time')

    mg_facecolor = '#99d7f2'
    plt.rcParams.update({
        'font.size': 10,
    })
    fig, axs = plt.subplots(MG, 2, figsize=(20, 8), gridspec_kw={'hspace': 0.3, 'wspace': 0.2})


    for mg in range(MG):
        axs[mg][1].text(x=.03, y=0.87, s=f'MG {mg}', transform=axs[mg][0].transAxes, bbox=dict(facecolor=mg_facecolor))
        axs[mg][0].plot(DA_data[f'Z from {mg}'], color='green', label='Day Ahead')
        axs[mg][0].plot(RT_data[f'Z from {mg}'], color='red', label='Real Time')

        axs[mg][1].text(x=.03, y=0.87, s=f'MG {mg}', transform=axs[mg][1].transAxes, bbox=dict(facecolor=mg_facecolor))
        axs[mg][1].plot(DA_data[f'V to {mg}'], color='green', label='Day Ahead')
        axs[mg][1].plot(RT_data[f'V to {mg}'], color='red', label='Real Time')

    for mg in range(MG):
        axs[mg][0].set_ylabel('Bid Amount Z (kW)')
        axs[mg][0].set_xlabel('Time Period t (hour)')
        axs[mg][0].set_xticks(range(len(DA_data)))

        axs[mg][1].set_ylabel('Allocated Power V (kW)')
        axs[mg][1].set_xlabel('Time Period t (hour)')
        axs[mg][1].set_xticks(range(len(DA_data)))

    axs[0][0].legend(loc=[0, 1.05])
    axs[0][1].legend(loc=[0, 1.05])

    return fig

