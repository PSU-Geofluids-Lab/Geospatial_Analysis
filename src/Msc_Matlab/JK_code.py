

% --- Target Parameters ---
tau_target = 0.4;
SE_target = 0.1;
Xf = 0.02;
noise_sigma = 0.12;
rho_soil = 1.3;
rho_feed = 2.7;
Xwf_target = (tau_target * Xf) / (1 - tau_target);
M_i = 47.867; M_j = 40.078;
soil_ppm.i = 12000; soil_ppm.j = 20000;
feed_ppm.i = 17000; feed_ppm.j = 80000;
soil.i = soil_ppm.i / (M_i * 1000); soil.j = soil_ppm.j / (M_j * 1000);
feedstock.i = feed_ppm.i / (M_i * 1000); feedstock.j = feed_ppm.j / (M_j * 1000);
pool_ICM = 35;
