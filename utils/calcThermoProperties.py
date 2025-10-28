import pandas as pd
import numpy as np
import os


def calcThermoProps(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate further thermodynamic properties based on the Helmholtz energy derivatives sampled by MD.
    Errors of the Helmholtz energy derivatives are required and must be calculated in advance.
    '''

    # Ideal parts
    df['A20_id'] = -0.5*df['dimension']

    # Isochoric heat capacity
    df['c_v'] = -df['A20_id']-df['A20r']
    df['c_v_err'] = df['A20r_err']  # Only A20r has an error

    # Isobaric heat capacity
    df['c_p'] = -(df['A20_id']+df['A20r'])+(((1+df['A01r']-df['A11r'])**2)/(1+2*df['A01r']+df['A02r']))
    # Partial derivatives for c_p error propagation
    term1 = (1+df['A01r']-df['A11r'])
    term2 = (1+2*df['A01r']+df['A02r'])
    term3 = 2 * term1 / term2
    term4 = -2 * term1**2 / term2**2
    dcp_dA20r = -1
    dcp_dA01r = term3 + term4
    dcp_dA11r = -2 * term1 / term2
    dcp_dA02r = -2 * term1**2 / term2**2
    df['c_p_err'] = np.sqrt(
        (dcp_dA20r * df['A20r_err'])**2 +
        (dcp_dA01r * df['A01r_err'])**2 +
        (dcp_dA11r * df['A11r_err'])**2 +
        (dcp_dA02r * df['A02r_err'])**2
    )
    
    # Speed of sound
    mass = 1.0
    df['speedofsound'] = ((df['Temp']/mass)*( 1+2*df['A01r']+df['A02r']-(((1+df['A01r']-df['A11r'])**2)/(df['A20_id']+df['A20r'])) ))**0.5
    # Partial derivatives for speed of sound error propagation
    term1 = (1+df['A01r']-df['A11r'])
    A20_total = df['A20_id']+df['A20r']
    dsd_dA01r = (df['Temp'] / df['speedofsound']) * (2 - (2 * term1 / A20_total))
    dsd_dA02r = (df['Temp'] / df['speedofsound'])
    dsd_dA11r = -(df['Temp'] / df['speedofsound']) * (2 * term1 / A20_total)
    dsd_dA20r = (df['Temp'] / df['speedofsound']) * (term1**2 / A20_total**2)
    df['speedofsound_err'] = 0.5*np.sqrt(
        (dsd_dA01r * df['A01r_err'])**2 +
        (dsd_dA02r * df['A02r_err'])**2 +
        (dsd_dA11r * df['A11r_err'])**2 +
        (dsd_dA20r * df['A20r_err'])**2
    )

    # Isothermal compressibility
    df['beta_T'] = 1/(df['rho']*df['Temp']*(1+df['A02r']+2*df['A01r']))
    # Partial derivatives for beta_T error propagation
    dbT_dA02r = -df['beta_T'] / (1+df['A02r']+2*df['A01r'])
    dbT_dA01r = -2 * df['beta_T'] / (1+df['A02r']+2*df['A01r'])
    df['beta_T_err'] = np.sqrt(
        (dbT_dA02r * df['A02r_err'])**2 +
        (dbT_dA01r * df['A01r_err'])**2
    )

    # Thermal pressure coefficient
    df['gamma_v'] = df['rho']*(1+df['A01r']-df['A11r'])
    # Partial derivatives for gamma_v error propagation
    dgv_dA01r = df['rho']
    dgv_dA11r = -df['rho']
    df['gamma_v_err'] = np.sqrt(
        (dgv_dA01r * df['A01r_err'])**2 +
        (dgv_dA11r * df['A11r_err'])**2
    )

    # Thermal expansion coefficient
    df['alpha'] = df['beta_T']*df['gamma_v']
    # Partial derivatives for alpha error propagation
    da_dbeta_T = df['gamma_v']
    da_dgamma_v = df['beta_T']
    df['alpha_err'] = np.sqrt(
        (da_dbeta_T * df['beta_T_err'])**2 +
        (da_dgamma_v * df['gamma_v_err'])**2
    )

    # Grueneisen parameter
    df['grueneisen'] = -(1+df['A01r']-df['A11r'])/(df['A20_id']+df['A20r'])
    # Partial derivatives for grueneisen error propagation
    dg_dA01r = -1 / (df['A20_id'] + df['A20r'])
    dg_dA11r = 1 / (df['A20_id'] + df['A20r'])
    dg_dA20r = -(1 + df['A01r'] - df['A11r']) / (df['A20_id'] + df['A20r'])**2
    df['grueneisen_err'] = np.sqrt(
        (dg_dA01r * df['A01r_err'])**2 +
        (dg_dA11r * df['A11r_err'])**2 +
        (dg_dA20r * df['A20r_err'])**2
    )

    # Density scaling exponent
    df['densScalExpo'] = -(df['A01r']-df['A11r'])/df['A20r']
    # Partial derivatives for densScalExpo error propagation
    ddse_dA01r = -1 / df['A20r']
    ddse_dA11r = 1 / df['A20r']
    ddse_dA20r = (df['A01r'] - df['A11r']) / (df['A20r']**2)
    df['densScalExpo_err'] = np.sqrt(
        (ddse_dA01r * df['A01r_err'])**2 +
        (ddse_dA11r * df['A11r_err'])**2 +
        (ddse_dA20r * df['A20r_err'])**2
    )

    return df

if __name__ == '__main__':

    # Path to csv file containing the processed simulation data (one line per simulation; including errors)
    path2Data = 'PATHTOCSVDATA'

    # Read csv file and calculate thermodynamic properties including their errors
    df = pd.read_csv(os.path.join(path2Data,'allData.dat'), index_col=0)
    df_TD = calcThermoProps(df)

    # Export
    df_TD.to_csv(os.path.join(path2Data,f'allData_TD.dat'), index=False)
