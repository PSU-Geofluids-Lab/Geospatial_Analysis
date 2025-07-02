import matplotlib.pylab as plt
import gc
import numpy as np
from scipy.optimize import minimize

# Zhe Z-score corresponding to the desired confidence level is calculated using a standard normal distribution (also known as a Z-distribution).
# The Z-distribution is a normal distribution with a mean of 0 and a standard deviation of 1. The Z-score is the number of standard deviations from the mean that a value is.
# 90% confidence: Z ≈ 1.65
# 95% confidence: Z ≈ 1.96
# 99% confidence: Z ≈ 2.58
# 99.9% confidence: Z ≈ 3.29

Z_score_dict = {
    90: 1.65,
    95: 1.96,
    99: 2.58,
    99.9: 3.29
}

def make_soil_samples_Pooling(measurement_error,  num_pooling_per_sample_unit, num_sample_units,num_actual_samples_per_sample_unit,
                              var_Ca, var_Ti, tau_target, soil_params, feed_params, weathered_params,added_var = 0.05):
    #% tau = Xwf / (Xwf + Xf)    
    #% Xwf = tau * Xf / (1 - tau)
    """
    Make a set of soil samples for the inversion with pooling.

    Parameters
    ----------
    measurement_error : float
        Measurement error in percent.
    num_pooling_per_sample_unit : int
        Number of pooling per sample unit.
    num_sample_units : int
        Number of sample units.
    num_actual_samples_per_sample_unit : int
        Number of actual samples per sample unit.
    var_Ca : float
        Variance of calcium in the soil.
    var_Ti : float
        Variance of titanium in the soil.
    tau_target : float
        Target value of tau.
    soil_params : dict
        Dictionary of soil parameters.
    feed_params : dict
        Dictionary of feedstock parameters.
    weathered_params : dict
        Dictionary of weathered parameters.
    added_var : float, optional
        Additional variance to add to the samples. Default is 0.05.

    Returns
    -------
    soil_j_noisy : ndarray
        Noisy values of j_mix.
    soil_i_noisy : ndarray
        Noisy values of i_mix.
    j_mix_noisy : ndarray
        Noisy values of j_mix.
    i_mix_noisy : ndarray
        Noisy values of i_mix.
    """
    print(f'Model with a feedstock volume fractions of {feed_params["Xf_init"]}')
    # Volume fraction of weathered material , We have specified the total feedstock volume fraction (Xf + Xwf) 
    Xwf_target = tau_target * feed_params['Xf_init'] 
    Xf = Xwf_target*(1 - tau_target)/tau_target; ## Volume fraction of feedstock 
    Xs = 1.0 -  feed_params['Xf_init']           ## Volume fraction of soil
    print(f'Model with params Xwf - {Xwf_target}, Xf - {Xf}')

    ## For every sample unit, we make num_pooling_per_sample_unit averaging & we have num_sample_units sample units
    random_vals_Ca = np.random.choice(soil_params['field_data_Ca'].flatten(), size=num_sample_units*num_pooling_per_sample_unit, replace=True)
    random_vals_Ti = np.random.choice(soil_params['field_data_Ti'].flatten(), size=num_sample_units*num_pooling_per_sample_unit, replace=True)
    soil_j = (random_vals_Ca*var_Ca + 1)*soil_params['soil_ppm_Ca_molg']
    soil_i = (random_vals_Ti*var_Ti + 1)*soil_params['soil_ppm_Ti_molg']
    feed_Ca =  feed_params['feed_ppm_Ca_molg']*(1+np.random.normal(0, feed_params['feed_ppm_Ca_molg_std'], num_sample_units*num_pooling_per_sample_unit))
    feed_Ti =  feed_params['feed_ppm_Ti_molg']*(1+np.random.normal(0, feed_params['feed_ppm_Ti_molg_std'], num_sample_units*num_pooling_per_sample_unit))
    ij_min_denom = Xs*soil_params['rho_soil'] + Xf*feed_params['rho_feed'] + Xwf_target*weathered_params['rho_weathered']
    j_mix_num = Xs*soil_params['rho_soil']*soil_j + \
                Xf*feed_params['rho_feed']*feed_Ca + \
                Xwf_target*weathered_params['rho_weathered']*weathered_params['soil_ppm_Ca_molg'] ## Because the feedstock weathered isn't random
    i_wf = soil_i + (feed_params['rho_feed']/weathered_params['rho_weathered']) * (feed_Ti - weathered_params['soil_ppm_Ti_molg'])
    i_mix_num = Xs*soil_params['rho_soil']*soil_i + \
                Xf*feed_params['rho_feed']*feed_Ti + \
                Xwf_target*weathered_params['rho_weathered']*i_wf
    j_mix_all = j_mix_num / ij_min_denom
    i_mix_all = i_mix_num / ij_min_denom

    j_mix     = j_mix_all.reshape(-1, num_pooling_per_sample_unit).mean(axis=1)
    j_mix     = (j_mix*(1 + np.random.normal(0, scale=added_var, size=(num_actual_samples_per_sample_unit, len(j_mix))))).flatten()
    # j_mix_std = j_mix_all.reshape(-1, num_pooling_per_sample_unit).std(axis=1)
    i_mix     = i_mix_all.reshape(-1, num_pooling_per_sample_unit).mean(axis=1)
    i_mix     = (i_mix*(1 + np.random.normal(0, scale=added_var, size=(num_actual_samples_per_sample_unit, len(i_mix))))).flatten()

    # Now setting up the soil reference for the inversion
    random_vals_Ca = np.random.choice(soil_params['field_data_Ca'].flatten(), size=num_sample_units*num_pooling_per_sample_unit, replace=True)
    random_vals_Ti = np.random.choice(soil_params['field_data_Ti'].flatten(), size=num_sample_units*num_pooling_per_sample_unit, replace=True)
    soil_j_all = (random_vals_Ca*var_Ca + 1)*soil_params['soil_ppm_Ca_molg']
    soil_i_all = (random_vals_Ti*var_Ti + 1)*soil_params['soil_ppm_Ti_molg']

    soil_j = soil_j_all.reshape(-1, num_pooling_per_sample_unit).mean(axis=1)
    soil_i = soil_i_all.reshape(-1, num_pooling_per_sample_unit).mean(axis=1)
    soil_j     = (soil_j*(1 + np.random.normal(0, scale=added_var, size=(num_actual_samples_per_sample_unit, len(soil_j))))).flatten()
    soil_i     = (soil_i*(1 + np.random.normal(0, scale=added_var, size=(num_actual_samples_per_sample_unit, len(soil_i))))).flatten()

    soil_j_err = soil_j * (measurement_error / 100)
    soil_i_err = soil_i * (measurement_error / 100)
    j_mix_err = j_mix * (measurement_error / 100)
    i_mix_err = i_mix * (measurement_error / 100)

    soil_j_noisy = soil_j + np.random.normal(0, soil_j_err, size=soil_j.shape)
    soil_i_noisy = soil_i + np.random.normal(0, soil_i_err, size=soil_i.shape)
    j_mix_noisy = j_mix + np.random.normal(0, j_mix_err, size=j_mix.shape)
    i_mix_noisy = i_mix + np.random.normal(0, i_mix_err, size=i_mix.shape)
    return soil_j_noisy,soil_i_noisy, j_mix_noisy, i_mix_noisy


def make_soil_samples(measurement_error,var_Ca, var_Ti, tau_target, num_samples, soil_params, feed_params, weathered_params):
    #% tau = Xwf / (Xwf + Xf)    
    #% Xwf = tau * Xf / (1 - tau)
    """
    Make a set of soil samples for the inversion.

    Parameters
    ----------
    var_feed : float
        Variance of feedstock compositions
    measurement_error : float
        Measurement error in percent.
    var_Ca : float
        Variance of calcium in the soil.
    var_Ti : float
        Variance of titanium in the soil.
    tau_target : float
        Target value of tau.
    num_samples : int
        Number of samples to generate.
    soil_params : dict
        Dictionary of soil parameters.
    feed_params : dict
        Dictionary of feedstock parameters.
    weathered_params : dict
        Dictionary of weathered parameters.

    Returns
    -------
    soil_j_noisy : ndarray
        Noisy values of j_mix.
    soil_i_noisy : ndarray
        Noisy values of i_mix.
    j_mix_noisy : ndarray
        Noisy values of j_mix.
    i_mix_noisy : ndarray
        Noisy values of i_mix.
    """
    print(f'Model with a feedstock volume fractions of {feed_params["Xf_init"]}')
    # Volume fraction of weathered material , We have specified the total feedstock volume fraction (Xf + Xwf) 
    Xwf_target = tau_target * feed_params['Xf_init'] 
    Xf = Xwf_target*(1 - tau_target)/tau_target; ## Volume fraction of feedstock 
    Xs = 1.0 -  feed_params['Xf_init']           ## Volume fraction of soil
    print(f'Model with params Xwf - {Xwf_target}, Xf - {Xf}')

    ## For every draw we have a soil reference for the mixing sample
    random_vals_Ca = np.random.choice(soil_params['field_data_Ca'].flatten(), size=num_samples, replace=True)
    random_vals_Ti = np.random.choice(soil_params['field_data_Ti'].flatten(), size=num_samples, replace=True)
    soil_j = (random_vals_Ca*var_Ca + 1)*soil_params['soil_ppm_Ca_molg']
    soil_i = (random_vals_Ti*var_Ti + 1)*soil_params['soil_ppm_Ti_molg']
    ij_min_denom = Xs*soil_params['rho_soil'] + Xf*feed_params['rho_feed'] + Xwf_target*weathered_params['rho_weathered']
    feed_Ca =  feed_params['feed_ppm_Ca_molg']*(1+np.random.normal(0, feed_params['feed_ppm_Ca_molg_std'], num_samples))
    feed_Ti =  feed_params['feed_ppm_Ti_molg']*(1+np.random.normal(0, feed_params['feed_ppm_Ti_molg_std'], num_samples))

    j_mix_num = Xs*soil_params['rho_soil']*soil_j + \
                Xf*feed_params['rho_feed']*feed_Ca + \
                Xwf_target*weathered_params['rho_weathered']*weathered_params['soil_ppm_Ca_molg'] ## Because the feedstock weathered isn't random
    i_wf = soil_i + (feed_params['rho_feed']/weathered_params['rho_weathered']) * (feed_Ti - weathered_params['soil_ppm_Ti_molg'])
    i_mix_num = Xs*soil_params['rho_soil']*soil_i + \
                Xf*feed_params['rho_feed']*feed_Ti + \
                Xwf_target*weathered_params['rho_weathered']*i_wf
    j_mix = j_mix_num / ij_min_denom
    i_mix = i_mix_num / ij_min_denom

    # Now setting up the soil reference for the inversion
    random_vals_Ca = np.random.choice(soil_params['field_data_Ca'].flatten(), size=num_samples, replace=True)
    random_vals_Ti = np.random.choice(soil_params['field_data_Ti'].flatten(), size=num_samples, replace=True)
    soil_j = (random_vals_Ca*var_Ca + 1)*soil_params['soil_ppm_Ca_molg']
    soil_i = (random_vals_Ti*var_Ti + 1)*soil_params['soil_ppm_Ti_molg']

    soil_j_err = soil_j * (measurement_error / 100)
    soil_i_err = soil_i * (measurement_error / 100)
    j_mix_err = j_mix * (measurement_error / 100)
    i_mix_err = i_mix * (measurement_error / 100)

    soil_j_noisy = soil_j + np.random.normal(0, soil_j_err, size=soil_j.shape)
    soil_i_noisy = soil_i + np.random.normal(0, soil_i_err, size=soil_i.shape)
    j_mix_noisy = j_mix + np.random.normal(0, j_mix_err, size=j_mix.shape)
    i_mix_noisy = i_mix + np.random.normal(0, i_mix_err, size=i_mix.shape)
    return soil_j_noisy,soil_i_noisy, j_mix_noisy, i_mix_noisy

def obj(inpt,params):
    """
    Objective function for the optimization problem.

    Parameters
    ----------
    inpt : numpy.ndarray
        An array containing the values of Xwf and tau to be optimized.
    params : tuple
        A tuple containing the observed i and j values, soil parameters,
        feedstock parameters, and weathered parameters.

    Returns
    -------
    float
        The negative log10 of the sum of the squared errors between the
        observed and predicted values of i and j.
    """
    
    Xwf,tau = inpt
    Xf = Xwf*(1 - tau)/tau
    i_obs, j_obs, soil_i, soil_j, soil_params, feed_params, weathered_params = params 
    Xs = 1.0 -  Xf - Xwf
    ij_min_denom = Xs*soil_params['rho_soil'] + Xf*feed_params['rho_feed'] + Xwf*weathered_params['rho_weathered']
    j_mix_num = Xs*soil_params['rho_soil']*soil_j + \
                Xf*feed_params['rho_feed']*feed_params['feed_ppm_Ca_molg'] + \
                Xwf*weathered_params['rho_weathered']*weathered_params['soil_ppm_Ca_molg'] ## Because the feedstock weathered isn't random
    i_wf = soil_i + (feed_params['rho_feed']/weathered_params['rho_weathered']) * (feed_params['feed_ppm_Ti_molg'] - soil_params['soil_ppm_Ti_molg'])
    i_mix_num = Xs*soil_params['rho_soil']*soil_i + \
                Xf*feed_params['rho_feed']*feed_params['feed_ppm_Ti_molg'] + \
                Xwf*weathered_params['rho_weathered']*i_wf
    j_mix = j_mix_num / ij_min_denom
    i_mix = i_mix_num / ij_min_denom
    err = (i_mix - i_obs)**2 + (j_mix - j_obs)**2
    return np.log10(err)

def estimate_tau_Xwf(params):
    """
    Estimate the optimal values of Xwf and tau using the Nelder-Mead optimization method.

    Args:
        params (tuple): A tuple containing the observed i and j values, soil parameters, 
                        feedstock parameters, and weathered parameters.

    Returns:
        numpy.ndarray: An array containing the estimated Xwf and tau values.
    """

    res = minimize(obj, x0=[0.005, 0.1], args=(params), method='Nelder-Mead', bounds=[(0.0001, 0.2), (0.01, 0.9)],tol=1e-6,options={'maxiter':1e5})
    return res.x

def estimate_tau_Xwf_BruteForce(Tau_test_all,X_wf_test_all,params):
    """
    Perform a brute-force minimization of the objective function to estimate the optimal Xwf and tau values.

    Args:
        Tau_test_all (numpy.ndarray): Array of tau values to test.
        X_wf_test_all (numpy.ndarray): Array of Xwf values to test.
        params (list): List of parameters to pass to the objective function.

    Returns:
        tuple: The optimal tau and Xwf values, and the minimum value of the objective function.
    """
    res = obj((X_wf_test_all,Tau_test_all),params)
    min_indx = np.argmin(res)    
    return Tau_test_all[min_indx],X_wf_test_all[min_indx],np.min(res)

def plot_results(Xf, Xwf,tau_estimate, cum_avg_Xf, cum_avg_Xwf,cum_avg_tau, cum_std_Xf, cum_std_Xwf,cum_std_tau):
    """
    Plot the results of the analysis including cumulative averages, cumulative standard deviations, and histograms.

    Args:
        Xf (numpy.ndarray): Array of filtered Xf values.
        Xwf (numpy.ndarray): Array of filtered Xwf values.
        tau_estimate (numpy.ndarray): Array of estimated tau values.
        cum_avg_Xf (numpy.ndarray): Cumulative average of Xf.
        cum_avg_Xwf (numpy.ndarray): Cumulative average of Xwf.
        cum_avg_tau (numpy.ndarray): Cumulative average of tau.
        cum_std_Xf (numpy.ndarray): Cumulative standard deviation of Xf.
        cum_std_Xwf (numpy.ndarray): Cumulative standard deviation of Xwf.
        cum_std_tau (numpy.ndarray): Cumulative standard deviation of tau.

    Plots:
        - First figure:
            - Subplot 1: Cumulative averages of Xf and Xwf.
            - Subplot 2: Cumulative standard deviations of Xf and Xwf.
            - Subplot 3: Histogram of Xf and Xwf.
        - Second figure:
            - Subplot 1: Tau estimates and cumulative average of tau.
            - Subplot 2: Cumulative standard deviation of tau.
            - Subplot 3: Histogram of tau estimates.
    """

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(cum_avg_Xf, label='Xf')
    plt.plot(cum_avg_Xwf, label='Xwf')
    plt.xlabel('Sample Index')
    plt.ylabel('Cumulative Average')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(cum_std_Xf, label='Xf')
    plt.plot(cum_std_Xwf, label='Xwf')
    plt.xlabel('Sample Index')
    plt.ylabel('Cumulative 95% CI on mean')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist(Xf, bins=20, alpha=0.5, label='Xf')
    plt.hist(Xwf,bins=20,  alpha=0.5, label='Xwf')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(tau_estimate, 'o',alpha=0.25)
    plt.plot(cum_avg_tau, label='tau')
    plt.xlabel('Sample Index')
    plt.ylabel('Cumulative Average')
    plt.legend()
    plt.grid('on')

    plt.subplot(1, 3, 2)
    plt.plot(cum_std_tau, label='tau_estimate')
    plt.xlabel('Sample Index')
    plt.ylabel('Cumulative 95% CI on mean')
    plt.legend()
    plt.grid('on')

    plt.subplot(1, 3, 3)
    plt.hist(tau_estimate, bins=20, alpha=0.5, label='tau')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid('on')
    plt.tight_layout()
    plt.show()

def base_analysis_single(measurement_error,tau_target,var_Ca,var_Ti,num_samples,soil_params,feed_params,
                         weathered_params,use_std,filter_choice = 'threshold'):
    """
    Perform base analysis for estimating parameters from soil samples.

    Parameters
    ----------
    measurement_error : float
        Measurement error in percent.
    tau_target : float
        Target value of tau.
    var_Ca : float
        Variance of calcium in the soil.
    var_Ti : float
        Variance of titanium in the soil.
    num_samples : int
        Number of samples to generate.
    soil_params : dict
        Dictionary of soil parameters.
    feed_params : dict
        Dictionary of feedstock parameters.
    weathered_params : dict
        Dictionary of weathered parameters.
    use_std : bool
        Flag indicating whether to use standard error for confidence intervals.
    filter_choice : str, optional
        The filtering method to use; options include 'threshold', 
        'adaptive_threshold', 'grid', 'grid_conservative'. Defaults to 'threshold'.

    Returns
    -------
    tuple
        A tuple containing:
        - Xf (numpy.ndarray): Filtered Xf values.
        - Xwf (numpy.ndarray): Filtered Xwf values.
        - tau_estimate (numpy.ndarray): Estimated tau values.
        - cum_avg_Xf (numpy.ndarray): Cumulative average of Xf.
        - cum_avg_Xwf (numpy.ndarray): Cumulative average of Xwf.
        - cum_avg_tau (numpy.ndarray): Cumulative average of tau.
        - CI_cum_std_Xf (numpy.ndarray): Cumulative standard error for Xf.
        - CI_cum_std_Xwf (numpy.ndarray): Cumulative standard error for Xwf.
        - CI_cum_std_tau (numpy.ndarray): Cumulative standard error for tau.

    Raises
    ------
    ValueError
        If an unsupported filter choice is provided.
    """

    soil_j,soil_i, j_mix, i_mix = make_soil_samples(measurement_error,var_Ca, var_Ti, tau_target, num_samples, soil_params, feed_params, weathered_params)

    tau_estimate = np.zeros(num_samples)
    Xwf = np.zeros(num_samples)
    res_fit = np.zeros(num_samples)

    ###################################################
    Tau_vals_test = np.linspace(0.01,0.9,200)
    Xwf_vals_test = np.linspace(0.001,0.05,200)
    Tau_test_all,X_wf_test_all = np.meshgrid(Tau_vals_test,Xwf_vals_test)
    Tau_test_all = Tau_test_all.flatten()
    X_wf_test_all = X_wf_test_all.flatten()
    ###################################################
    for i in range(num_samples):
        params = [i_mix[i], j_mix[i], soil_i[i], soil_j[i], soil_params, feed_params, weathered_params]
        tau_estimate[i],Xwf[i],res_fit[i] = estimate_tau_Xwf_BruteForce(Tau_test_all,X_wf_test_all,params)
            
    Xf = Xwf*(1 - tau_estimate)/tau_estimate
    plt.figure(figsize=(5,5))
    plt.plot(res_fit)
    plt.axhline(y=0.7*np.min(res_fit), color='r', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('log Residual')
    plt.show()

    Xf = Xwf*(1 - tau_estimate)/tau_estimate
    if filter_choice == 'threshold':
        filter_indx = np.where(res_fit <= -7)[0]
    elif filter_choice == 'adaptive_threshold':
        filter_indx = np.where((res_fit <= 0.7*np.min(res_fit)))[0]
    elif filter_choice == 'grid':
        filter_indx = np.where((tau_estimate >= np.min(Tau_vals_test)) & 
                        (tau_estimate < np.max(Tau_vals_test)) & 
                        (Xwf >= np.min(Xwf_vals_test)) & 
                        (Xwf < np.max(Xwf_vals_test)))[0]
    elif filter_choice == 'grid_conservative':
        filter_indx = np.where((tau_estimate > np.min(Tau_vals_test)) & 
                        (tau_estimate < np.max(Tau_vals_test)) & 
                        (Xwf > np.min(Xwf_vals_test)) & 
                        (Xwf < np.max(Xwf_vals_test)))[0]
    else:
        raise ValueError(f"Unsupported filter choice: {filter_choice}, must be one of ['threshold', 'adaptive_threshold', 'grid', 'grid_conservative']")

    # assuming Xf and Xwf are numpy arrays
    cum_avg_Xf = np.cumsum(Xf) / np.arange(1, len(Xf) + 1)
    cum_avg_Xwf = np.cumsum(Xwf) / np.arange(1, len(Xwf) + 1)

    cum_std_Xf = np.array([np.std(Xf[:i+1]) for i in range(len(Xf))])
    cum_std_Xwf = np.array([np.std(Xwf[:i+1]) for i in range(len(Xwf))])
    cum_avg_tau = np.cumsum(tau_estimate) / np.arange(1, len(tau_estimate) + 1)
    cum_std_tau = np.array([np.std(tau_estimate[:i+1]) for i in range(len(tau_estimate))])
    if use_std :
        CI_cum_std_Xf = 1.96*cum_std_Xf/np.sqrt(len(Xf)) # Standard error of the mean
        CI_cum_std_Xwf = 1.96*cum_std_Xwf/np.sqrt(len(Xwf)) # Standard error of the mean
        CI_cum_std_tau = 1.96*cum_std_tau/np.sqrt(len(tau_estimate)) # Standard error of the mean
    else : 
        CI_cum_std_Xf = cum_std_Xf 
        CI_cum_std_Xwf = cum_std_Xwf 
        CI_cum_std_tau = cum_std_tau 

    plot_results(Xf, Xwf,tau_estimate, cum_avg_Xf, cum_avg_Xwf,cum_avg_tau, CI_cum_std_Xf, CI_cum_std_Xwf,CI_cum_std_tau)

    print('Filtered Results')
    print('Filtered number of samples: ', len(filter_indx))
    Xwf_filtered = Xwf[filter_indx]
    tau_estimate_filtered = tau_estimate[filter_indx]
    Xf_filtered = Xwf_filtered*(1 - tau_estimate_filtered)/tau_estimate_filtered
    cum_avg_Xf_filtered = np.cumsum(Xf_filtered) / np.arange(1, len(Xf_filtered) + 1)
    cum_avg_Xwf_filtered = np.cumsum(Xwf_filtered) / np.arange(1, len(Xwf_filtered) + 1)

    cum_std_Xf_filtered = np.array([np.std(Xf_filtered[:i+1]) for i in range(len(Xf_filtered))])
    cum_std_Xwf_filtered = np.array([np.std(Xwf_filtered[:i+1]) for i in range(len(Xwf_filtered))])
    cum_avg_tau_filtered = np.cumsum(tau_estimate_filtered) / np.arange(1, len(tau_estimate_filtered) + 1)
    cum_std_tau_filtered = np.array([np.std(tau_estimate_filtered[:i+1]) for i in range(len(tau_estimate_filtered))])
    # CI_95 = 1.96 * SEM
    if use_std :
        CI_cum_std_Xf_filtered = 1.96*cum_std_Xf_filtered/np.sqrt(len(Xf_filtered)) # Standard error of the mean
        CI_cum_std_Xwf_filtered = 1.96*cum_std_Xwf_filtered/np.sqrt(len(Xwf_filtered)) # Standard error of the mean
        CI_cum_std_tau_filtered = 1.96*cum_std_tau_filtered/np.sqrt(len(tau_estimate_filtered)) # Standard error of the mean
    else : 
        CI_cum_std_Xf_filtered = cum_std_Xf_filtered
        CI_cum_std_Xwf_filtered = cum_std_Xwf_filtered
        CI_cum_std_tau_filtered = cum_std_tau_filtered

    plot_results(Xf_filtered, Xwf_filtered,tau_estimate_filtered, cum_avg_Xf_filtered, cum_avg_Xwf_filtered,cum_avg_tau_filtered, 
                 CI_cum_std_Xf_filtered, CI_cum_std_Xwf_filtered,CI_cum_std_tau_filtered)
    plt.figure(figsize=(8, 8))
    plt.plot(filter_indx)
    plt.xlabel('Filtered Sample Index')
    plt.ylabel('Filtering Index')
    plt.grid('on')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.title('Results (all) - Symbol Size related to fit accuracy')
    im = plt.scatter(tau_estimate,Xwf,c=Xf,s=res_fit**2.,cmap='RdYlBu_r')
    plt.xlabel('tau estimate')
    plt.ylabel('Xwf')
    plt.colorbar(im, label='Xf')
    results = {
        'cum_avg_Xf': cum_avg_Xf,
        'cum_avg_Xwf': cum_avg_Xwf,
        'cum_avg_tau': cum_avg_tau,
        'CI_cum_std_Xf': CI_cum_std_Xf,
        'CI_cum_std_Xwf': CI_cum_std_Xwf,
        'CI_cum_std_tau': CI_cum_std_tau
    }

    results_filtered = {
        'cum_avg_Xf': cum_avg_Xf_filtered,
        'cum_avg_Xwf': cum_avg_Xwf_filtered,
        'cum_avg_tau': cum_avg_tau_filtered,
        'CI_cum_std_Xf': CI_cum_std_Xf_filtered,
        'CI_cum_std_Xwf': CI_cum_std_Xwf_filtered,
        'CI_cum_std_tau': CI_cum_std_tau_filtered,
        'filter_indx':filter_indx
    }
    return Xf, Xwf,tau_estimate, results,results_filtered

def base_analysis_pool(measurement_error,tau_target,var_Ca,var_Ti,soil_params,feed_params,weathered_params,
                       num_pooling_per_sample_unit,num_sample_units,num_actual_samples_per_sample_unit,use_std,filter_choice='threshold'):

    """
    Perform the base analysis for estimating parameters from soil samples with pooling.

    Parameters
    ----------
    measurement_error : float
        Measurement error in percent.
    tau_target : float
        Target value of tau.
    var_Ca : float
        Variance of calcium in the soil.
    var_Ti : float
        Variance of titanium in the soil.
    num_pooling_per_sample_unit : int
        Number of times to pool samples.
    num_sample_units : int
        Number of sample units.
    num_actual_samples_per_sample_unit : int
        Number of actual samples per sample unit.
    soil_params : dict
        Dictionary of soil parameters.
    feed_params : dict
        Dictionary of feedstock parameters.
    weathered_params : dict
        Dictionary of weathered parameters.
    use_std : bool
        Flag indicating whether to use standard error for confidence intervals.
    filter_choice : str, optional
        The filtering method to use; options include 'threshold', 'adaptive_threshold', 'grid', 'grid_conservative'. Defaults to 'threshold'.

    Returns
    -------
    tuple
        A tuple containing:
        - Xf (numpy.ndarray): Filtered Xf values.
        - Xwf (numpy.ndarray): Filtered Xwf values.
        - tau_estimate (numpy.ndarray): Estimated tau values.
        - cum_avg_Xf (numpy.ndarray): Cumulative average of Xf.
        - cum_avg_Xwf (numpy.ndarray): Cumulative average of Xwf.
        - cum_avg_tau (numpy.ndarray): Cumulative average of tau.
        - CI_cum_std_Xf (numpy.ndarray): Cumulative standard error for Xf.
        - CI_cum_std_Xwf (numpy.ndarray): Cumulative standard error for Xwf.
        - CI_cum_std_tau (numpy.ndarray): Cumulative standard error for tau.

    Raises
    ------
    ValueError
        If an unsupported filter choice is provided.
    """
    soil_j,soil_i, j_mix, i_mix = make_soil_samples_Pooling(measurement_error,  num_pooling_per_sample_unit, num_sample_units,
                                                            num_actual_samples_per_sample_unit,
                              var_Ca, var_Ti, tau_target, soil_params, feed_params, weathered_params)

    tau_estimate = np.zeros(num_sample_units*num_actual_samples_per_sample_unit)
    Xwf = np.zeros(num_sample_units*num_actual_samples_per_sample_unit)
    res_fit = np.zeros(num_sample_units*num_actual_samples_per_sample_unit)

    ###################################################
    Tau_vals_test = np.linspace(0.01,0.9,200)
    Xwf_vals_test = np.linspace(0.001,0.05,200)
    Tau_test_all,X_wf_test_all = np.meshgrid(Tau_vals_test,Xwf_vals_test)
    Tau_test_all = Tau_test_all.flatten()
    X_wf_test_all = X_wf_test_all.flatten()
    ###################################################
    for i in range(num_sample_units*num_actual_samples_per_sample_unit):
        params = [i_mix[i], j_mix[i], soil_i[i], soil_j[i], soil_params, feed_params, weathered_params]
        tau_estimate[i],Xwf[i],res_fit[i] = estimate_tau_Xwf_BruteForce(Tau_test_all,X_wf_test_all,params)
    
    plt.figure(figsize=(5,5))
    plt.plot(res_fit)
    plt.axhline(y=0.7*np.min(res_fit), color='r', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('log Residual')
    plt.show()

    Xf = Xwf*(1 - tau_estimate)/tau_estimate
    if filter_choice == 'threshold':
        filter_indx = np.where(res_fit <= -7)[0]
    elif filter_choice == 'adaptive_threshold':
        filter_indx = np.where((res_fit <= 0.7*np.min(res_fit)))[0]
    elif filter_choice == 'grid':
        filter_indx = np.where((tau_estimate >= np.min(Tau_vals_test)) & 
                        (tau_estimate < np.max(Tau_vals_test)) & 
                        (Xwf >= np.min(Xwf_vals_test)) & 
                        (Xwf < np.max(Xwf_vals_test)))[0]
    elif filter_choice == 'grid_conservative':
        filter_indx = np.where((tau_estimate > np.min(Tau_vals_test)) & 
                        (tau_estimate < np.max(Tau_vals_test)) & 
                        (Xwf > np.min(Xwf_vals_test)) & 
                        (Xwf < np.max(Xwf_vals_test)))[0]
    else:
        raise ValueError(f"Unsupported filter choice: {filter_choice}, must be one of ['threshold', 'adaptive_threshold', 'grid', 'grid_conservative']")

    # assuming Xf and Xwf are numpy arrays
    cum_avg_Xf = np.cumsum(Xf) / np.arange(1, len(Xf) + 1)
    cum_avg_Xwf = np.cumsum(Xwf) / np.arange(1, len(Xwf) + 1)

    cum_std_Xf = np.array([np.std(Xf[:i+1]) for i in range(len(Xf))])
    cum_std_Xwf = np.array([np.std(Xwf[:i+1]) for i in range(len(Xwf))])
    cum_avg_tau = np.cumsum(tau_estimate) / np.arange(1, len(tau_estimate) + 1)
    cum_std_tau = np.array([np.std(tau_estimate[:i+1]) for i in range(len(tau_estimate))])

    if use_std :
        CI_cum_std_Xf = 1.96*cum_std_Xf/np.sqrt(len(Xf)) # Standard error of the mean
        CI_cum_std_Xwf = 1.96*cum_std_Xwf/np.sqrt(len(Xwf)) # Standard error of the mean
        CI_cum_std_tau = 1.96*cum_std_tau/np.sqrt(len(tau_estimate)) # Standard error of the mean
    else : 
        CI_cum_std_Xf = cum_std_Xf 
        CI_cum_std_Xwf = cum_std_Xwf 
        CI_cum_std_tau = cum_std_tau 

    plot_results(Xf, Xwf,tau_estimate, cum_avg_Xf, cum_avg_Xwf,cum_avg_tau, CI_cum_std_Xf, CI_cum_std_Xwf,CI_cum_std_tau)

    print('Filtered Results')
    print('Filtered number of samples: ', len(filter_indx))
    Xwf_filtered = Xwf[filter_indx]
    tau_estimate_filtered = tau_estimate[filter_indx]
    Xf_filtered = Xwf_filtered*(1 - tau_estimate_filtered)/tau_estimate_filtered
    cum_avg_Xf_filtered = np.cumsum(Xf_filtered) / np.arange(1, len(Xf_filtered) + 1)
    cum_avg_Xwf_filtered = np.cumsum(Xwf_filtered) / np.arange(1, len(Xwf_filtered) + 1)

    cum_std_Xf_filtered = np.array([np.std(Xf_filtered[:i+1]) for i in range(len(Xf_filtered))])
    cum_std_Xwf_filtered = np.array([np.std(Xwf_filtered[:i+1]) for i in range(len(Xwf_filtered))])
    cum_avg_tau_filtered = np.cumsum(tau_estimate_filtered) / np.arange(1, len(tau_estimate_filtered) + 1)
    cum_std_tau_filtered = np.array([np.std(tau_estimate_filtered[:i+1]) for i in range(len(tau_estimate_filtered))])

    # CI_95 = 1.96 * SEM
    if use_std :
        CI_cum_std_Xf_filtered = 1.96*cum_std_Xf_filtered/np.sqrt(len(Xf_filtered)) # Standard error of the mean
        CI_cum_std_Xwf_filtered = 1.96*cum_std_Xwf_filtered/np.sqrt(len(Xwf_filtered)) # Standard error of the mean
        CI_cum_std_tau_filtered = 1.96*cum_std_tau_filtered/np.sqrt(len(tau_estimate_filtered)) # Standard error of the mean
    else : 
        CI_cum_std_Xf_filtered = cum_std_Xf_filtered
        CI_cum_std_Xwf_filtered = cum_std_Xwf_filtered
        CI_cum_std_tau_filtered = cum_std_tau_filtered

    plot_results(Xf_filtered, Xwf_filtered,tau_estimate_filtered, cum_avg_Xf_filtered, cum_avg_Xwf_filtered,cum_avg_tau_filtered, 
                 CI_cum_std_Xf_filtered, CI_cum_std_Xwf_filtered,CI_cum_std_tau_filtered)
    plt.figure(figsize=(8, 8))
    plt.plot(filter_indx)
    plt.xlabel('Filtered Sample Index')
    plt.ylabel('Filtering Index')
    plt.grid('on')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.title('Results (all) - Symbol Size related to fit accuracy')
    im = plt.scatter(tau_estimate,Xwf,c=Xf,s=res_fit**2.,cmap='RdYlBu_r')
    plt.xlabel('tau estimate')
    plt.ylabel('Xwf')
    plt.colorbar(im, label='Xf')
    results = {
        'cum_avg_Xf': cum_avg_Xf,
        'cum_avg_Xwf': cum_avg_Xwf,
        'cum_avg_tau': cum_avg_tau,
        'CI_cum_std_Xf': CI_cum_std_Xf,
        'CI_cum_std_Xwf': CI_cum_std_Xwf,
        'CI_cum_std_tau': CI_cum_std_tau
    }

    results_filtered = {
        'cum_avg_Xf': cum_avg_Xf_filtered,
        'cum_avg_Xwf': cum_avg_Xwf_filtered,
        'cum_avg_tau': cum_avg_tau_filtered,
        'CI_cum_std_Xf': CI_cum_std_Xf_filtered,
        'CI_cum_std_Xwf': CI_cum_std_Xwf_filtered,
        'CI_cum_std_tau': CI_cum_std_tau_filtered,
        'filter_indx':filter_indx
    }
    return Xf, Xwf,tau_estimate, results,results_filtered

def cost_analysis(cum_avg_tau,cum_std_tau,feed_params,
                  Integrated_application_Tons_basalt= 1000.0,
                  dollar_per_sample=50.0,price_ton=300.0,filter_indx = None, use_filter_indx = True):
    # dollar_per_sample = 50.0
    # price_ton = 300.0
    # Integrated_application_Tons_basalt = 10000.0

    #CO2 = 3.62 Mg + 2.2 Ca + 1.91 Na + 1.12 K (in kg for each component)
    # convert ton to kg, and ppm (1 million - 1e6 to kg/kg) 
    """
    Calculate and plot the costs of MRV and the income from the sequensted CO2

    Parameters
    ----------
    cum_avg_tau : numpy.ndarray
        Cumulative average of tau
    cum_std_tau : numpy.ndarray
        Cumulative standard deviation of tau
    feed_params : dict
        Parameters of the feedstock (basalt)
    Integrated_application_Tons_basalt : float, optional
        Integrated application in tons of basalt, defaults to 1000.0
    dollar_per_sample : float, optional
        Dollar cost per sample, defaults to 50.0
    price_ton : float, optional
        Price per ton of CO2, defaults to 300.0
    filter_indx : numpy.ndarray, optional
        Sample indices to filter on, defaults to None
    use_filter_indx : bool, optional
        Whether to use the filter indices, defaults to True

    Returns
    -------
    income : numpy.ndarray
        Income from sequenced CO2
    MRV_costs : numpy.ndarray
        Costs of MRV
    """
    Ca_mass_kg = feed_params['feed_ppm_Ca']*1e-6*Integrated_application_Tons_basalt*1e3  
    CO2_seq = 2.2*Ca_mass_kg
    

    if use_filter_indx:
        sample_cnt = filter_indx.copy()
    else: 
        sample_cnt = np.arange(1, len(cum_avg_tau) + 1)

    income = CO2_seq*(cum_avg_tau-cum_std_tau)*price_ton/1e3
    MRV_costs = dollar_per_sample*sample_cnt

    fig, axs = plt.subplots(2, figsize=(10, 6))

    axs[0].plot(sample_cnt,MRV_costs, label='MRV Costs')
    axs[0].plot(sample_cnt,income, label='Income')
    axs[0].legend()
    axs[0].set_title('MRV Costs and Income')
    axs[0].set_xlabel('Number of samples')
    axs[0].set_ylabel('Value (USD)')
    axs[0].grid('on')

    axs[1].plot(sample_cnt,income - MRV_costs)
    axs[1].set_title('Net Income (Income - MRV Costs)')
    axs[1].set_xlabel('Number of samples')
    axs[1].set_ylabel('Net Value (USD)')
    plt.grid('on')

    plt.tight_layout()
    plt.show()
    return income, MRV_costs