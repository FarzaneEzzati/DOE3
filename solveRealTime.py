'''This file contains a function solving real time models and returns optimized decisions'''
import pandas as pd

from buildCEMS import buildDACEMS, buildRTCEMS  # import functions to build models for individual MGs
from buildMG import buildDAMG, buildRTMG  # import functions to build models for CEMS
import pickle as pkl


def solveRT(mgi, cems, SH, PH, CH):
    mgm_rt = {i: buildRTMG(mgi[i], PH) for i in mgi.keys()}
    cemsm_rt = buildRTCEMS(mgi, cems, PH)

    I_range = range(len(mgi.keys()))
    T_range = range(PH)
    # Set initial ES level for MGS and CEMS
    E0_mg = {i: mgi[i].dv['es'] for i in mgi.keys()}
    E0_cems = cems.dv['es']

    Z_rt = {i: {} for i in mgm_rt.keys()} # These are temporary
    Y_rt = {i: {} for i in mgm_rt.keys()}

    mg_Z_rt = {i: {} for i in mgm_rt.keys()}  # Dictionary to save real time Z
    mg_Y_rt = {i: {} for i in mgm_rt.keys()}  # Dictionary to save Ymax
    mg_Obj_rt = {i: {} for i in mgm_rt.keys()}  # Dictionary to save real time deviation

    cems_V_rt = {i: {} for i in mgm_rt.keys()}  # Dictionary to save real time V
    cems_Obj_rt = {}  # Dictionary to save real time deviation
    # Load day ahead solutions for MGs
    with open('results/dayAheadMGs.pkl', 'rb') as handle:
        mg_Ymax, mg_Lmax, mg_Z, mg_Xes, mg_Xpv, mg_Xdg = pkl.load(handle)
    handle.close()
    # Load day ahead solutions for CEMS
    with open('results/dayAheadCEMS.pkl', 'rb') as handle:
        V, R, K, _EA, _ER, _ES = pkl.load(handle)
    handle.close()

    t_start = 0  # Starting real time optimization from time 0
    while t_start + PH <= SH:
        save_range = range(PH) if t_start == SH - PH else range(CH)
        for i in mgm_rt.keys():
            # Get prediction data
            LPrediction = pd.read_csv('systemInfo/LoadRTMGs.csv')[f'{i}'].iloc[t_start:t_start + PH].values
            PVPrediction = pd.read_csv('systemInfo/PVRTMGs.csv')[f'{i}'].iloc[t_start:t_start + PH].values
            # Update RT models
            mgm_rt[i].setAttr('RHS', mgm_rt[i].getConstrByName('E0'), E0_mg[i])
            for t in range(PH):
                mgm_rt[i].setAttr('RHS', mgm_rt[i].getConstrByName(f'LoadPred[{t}]'), LPrediction[t])
                mgm_rt[i].setAttr('RHS', mgm_rt[i].getConstrByName(f'PVPred[{t}]'), PVPrediction[t])
                mgm_rt[i].setAttr('RHS', mgm_rt[i].getConstrByName(f'ZPred[{t}]'), mg_Z[i][t])
                mgm_rt[i].setAttr('RHS', mgm_rt[i].getConstrByName(f'VElements[{t}]'), V[i][t])
            mgm_rt[i].update()

            # Solve RT model
            mgm_rt[i].optimize()

            # Get optimal decision for the CH
            for t in save_range:
                Z_rt[i][t] = mgm_rt[i].getVarByName(f'Z[{t}]').x
                Y_rt[i][t] = mgm_rt[i].getVarByName(f'Y[{t}]').x
                mg_Obj_rt[i][t_start + t] = mgm_rt[i].ObjVal / mgi[i].gamma['sdp']
                mg_Z_rt[i][t_start + t] = Z_rt[i][t]
                mg_Y_rt[i][t_start + t] = Y_rt[i][t]


            # Update E0 for the next time span
            E0_mg[i] = mgm_rt[i].getVarByName(f'E[{PH}]').x

        # Prepare predicted data for CEMS
        PVPredictionCEMS = pd.read_csv('systemInfo/PVRTCEMS.csv')['pv'].iloc[t_start:t_start + PH].values

        # Solve real time for CEMS
        for mg in I_range:
            for t in save_range:
                cemsm_rt.setAttr('RHS', cemsm_rt.getConstrByName(f'Vmax[{mg},{t}]'), mg_Ymax[mg][t])
                cemsm_rt.setAttr('RHS', cemsm_rt.getConstrByName(f'VPred[{mg},{t}]'), V[mg][t])
                cemsm_rt.setAttr('RHS', cemsm_rt.getConstrByName(f'Koutput[{mg},{t}]'), mg_Z[mg][t])
        for t in T_range:
            cemsm_rt.setAttr('RHS', cemsm_rt.getConstrByName(f'PVPred[{t}]'), PVPredictionCEMS[t])
            cemsm_rt.setAttr('RHS', cemsm_rt.getConstrByName('E0'), E0_cems)

        # Solve RT model
        cemsm_rt.update()
        cemsm_rt.optimize()

        # Save real time V
        for t in save_range:
            cems_Obj_rt[t_start + t] = cemsm_rt.ObjVal / cems.gamma['adp']
            for mg in I_range:
                cems_V_rt[mg][t_start + t] = cemsm_rt.getVarByName(f'V[{mg},{t}]').x

        # Update E0 for the next time span
        E0_cems = cemsm_rt.getVarByName(f'E[{PH}]').x

        # Update start time before end while
        t_start += CH


    # Save real time results for reporting
    with open('results/realTimeMGs.pkl', 'wb') as handle:
        pkl.dump([mg_Obj_rt, mg_Z_rt, mg_Y_rt], handle)
    handle.close()
    with open('results/realTimeCEMS.pkl', 'wb') as handle:
        pkl.dump([cems_Obj_rt, cems_V_rt], handle)
    handle.close()


