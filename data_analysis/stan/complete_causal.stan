data {
  int<lower=1> N; // number of observations
  
  // categorical variables
  int<lower=1> K_D; // datasets
  int<lower=1> K_P; // preprocessing types
  int<lower=1> K_E; // news encoders
  int<lower=1> K_L; // text scopes
  int<lower=1> K_U; // user encoders
  int<lower=1> K_G; // similarity functions
  int<lower=1> K_W; // evaluation windows
  int<lower=1> K_WW; // weekend/weekday
  int<lower=1> K_ToD; // time of day bins
  int<lower=1> K_R; // random seeds
  
  // Categorical variable indices
  array[N] int<lower=1, upper=K_D> D_id;
  array[N] int<lower=1, upper=K_P> P_id;
  array[N] int<lower=1, upper=K_E> E_id;
  array[N] int<lower=1, upper=K_L> L_id;
  array[N] int<lower=1, upper=K_U> U_id;
  array[N] int<lower=1, upper=K_G> G_id;
  array[N] int<lower=1, upper=K_W> W_id;
  array[N] int<lower=1, upper=K_WW> WW_id;
  array[N] int<lower=1, upper=K_ToD> ToD_id;
  array[N] int<lower=1, upper=K_R> R_id;
  
  // contextual variables
  vector[N] CS_user; // user cold-start ratio, standardized
  vector[N] CS_item; // item cold-start ratio, standardized
  vector[N] UA; // user activity, standardized
  
  // Observed metric
  vector<lower=0, upper=1>[N] Y;
  
  // number of impressions used to compute each metric
  vector<lower=1>[N] n_impressions;
  
  // Small value for logit transform
  real<lower=0> epsilon;
}
transformed data {
  vector[N] Y_logit;
  vector[N] obs_scale;
  real mean_n;
  real mean_Y_logit;
  
  mean_n = mean(n_impressions);
  
  for (i in 1 : N) {
    real y_clipped;
    y_clipped = fmin(1 - epsilon, fmax(epsilon, Y[i]));
    Y_logit[i] = logit(y_clipped);
    obs_scale[i] = sqrt(mean_n / n_impressions[i]);
  }
  mean_Y_logit = mean(Y_logit);
}
parameters {
  real alpha;
  
  // Main effects
  sum_to_zero_vector[K_D] a_D;
  sum_to_zero_vector[K_P] a_P;
  sum_to_zero_vector[K_E] a_E;
  sum_to_zero_vector[K_G] a_G;
  sum_to_zero_vector[K_W] a_W;
  sum_to_zero_vector[K_U] a_U_raw;
  sum_to_zero_vector[K_L] a_L_raw;
  sum_to_zero_vector[K_R] a_R_raw;
  
  // Scales for partial pooling
  real<lower=1e-4, upper=2> sigma_D;
  real<lower=1e-4, upper=2> sigma_P;
  real<lower=1e-4, upper=2> sigma_E;
  real<lower=1e-4, upper=2> sigma_L;
  real<lower=1e-4, upper=2> sigma_U;
  real<lower=1e-4, upper=2> sigma_G;
  real<lower=1e-4, upper=2> sigma_W;
  real<lower=1e-4, upper=2> sigma_R;
  
  // Continuous covariate effects
  real beta_CS_user;
  real beta_CS_item;
  real beta_UA;
  
  // Observation noise
  real<lower=1e-4, upper=1> sigma_obs;
}
transformed parameters {
  vector[K_U] a_U = sigma_U * a_U_raw;
  vector[K_L] a_L = sigma_L * a_L_raw;
  vector[K_R] a_R = sigma_R * a_R_raw;
}
model {
  vector[N] mu;
  
  // Priors
  alpha ~ normal(mean_Y_logit, 1);
  
  a_D ~ normal(0, sigma_D);
  a_P ~ normal(0, sigma_P);
  a_E ~ normal(0, sigma_E);
  a_G ~ normal(0, sigma_G);
  a_W ~ normal(0, sigma_W);
  a_U_raw ~ normal(0, 1);
  a_L_raw ~ normal(0, 1);
  a_R_raw ~ normal(0, 1);
  
  sigma_D ~ normal(0, 0.5);
  sigma_P ~ normal(0, 0.5);
  sigma_E ~ normal(0, 0.5);
  sigma_L ~ normal(0, 0.5);
  sigma_U ~ normal(0, 0.5);
  sigma_G ~ normal(0, 0.5);
  sigma_W ~ normal(0, 0.5);
  sigma_R ~ normal(0, 0.5);
  
  beta_CS_user ~ normal(0, 0.5);
  beta_CS_item ~ normal(0, 0.5);
  beta_UA ~ normal(0, 0.5);
  
  sigma_obs ~ normal(0, 0.5);
  
  for (i in 1 : N) {
    mu[i] = alpha + a_D[D_id[i]] + a_P[P_id[i]] + a_E[E_id[i]] + a_L[L_id[i]]
            + a_U[U_id[i]] + a_G[G_id[i]] + a_W[W_id[i]] + a_R[R_id[i]]
            + beta_CS_user * CS_user[i] + beta_CS_item * CS_item[i]
            + beta_UA * UA[i];
    
    Y_logit[i] ~ normal(mu[i], sigma_obs * obs_scale[i]);
  }
}
generated quantities {
  vector[N] mu;
  vector[N] Y_hat;
  vector[N] log_lik;
  
  for (i in 1 : N) {
    real sigma_i;
    
    mu[i] = alpha + a_D[D_id[i]] + a_P[P_id[i]] + a_E[E_id[i]] + a_L[L_id[i]]
            + a_U[U_id[i]] + a_G[G_id[i]] + a_W[W_id[i]] + a_R[R_id[i]]
            + beta_CS_user * CS_user[i] + beta_CS_item * CS_item[i]
            + beta_UA * UA[i];
    
    sigma_i = sigma_obs * obs_scale[i];
    
    Y_hat[i] = inv_logit(mu[i]);
    log_lik[i] = normal_lpdf(Y_logit[i] | mu[i], sigma_i);
  }
}
