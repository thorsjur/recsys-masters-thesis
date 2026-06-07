data {
  int<lower=1> N; // number of observations
  
  // categorical variables
  int<lower=1> K_D;   // datasets
  int<lower=1> K_E;   // news encoders
  int<lower=1> K_L;   // text scopes
  int<lower=1> K_U;   // user encoders
  int<lower=1> K_G;   // similarity functions
  int<lower=1> K_W;   // evaluation windows
  int<lower=1> K_R;   // random seeds
  
  // categorical variable indices
  array[N] int<lower=1, upper=K_D> D_id;
  array[N] int<lower=1, upper=K_E> E_id;
  array[N] int<lower=1, upper=K_L> L_id;
  array[N] int<lower=1, upper=K_U> U_id;
  array[N] int<lower=1, upper=K_G> G_id;
  array[N] int<lower=1, upper=K_W> W_id;
  array[N] int<lower=1, upper=K_R> R_id;
  
  // contextual variables
  vector[N] CS_user;
  vector[N] CS_item;
  vector[N] UA;
  
  // observed metric
  vector<lower=0, upper=1>[N] Y;
  
  // number of impressions used to compute each metric
  vector<lower=1>[N] n_impressions;
  
  real<lower=0> epsilon;
}

transformed data {
  vector[N] Y_logit;
  vector[N] obs_scale;
  real mean_n;
  real mean_Y_logit;
  
  mean_n = mean(n_impressions);
  
  for (i in 1:N) {
    real y_clipped;
    y_clipped = fmin(1 - epsilon, fmax(epsilon, Y[i]));
    Y_logit[i] = logit(y_clipped);
    obs_scale[i] = sqrt(mean_n / n_impressions[i]);
  }
  
  mean_Y_logit = mean(Y_logit);
}

parameters {
  real alpha;
  
  // Non-centered main effects
  sum_to_zero_vector[K_D] a_D_raw;
  sum_to_zero_vector[K_E] a_E_raw;
  sum_to_zero_vector[K_G] a_G_raw;
  sum_to_zero_vector[K_U] a_U_raw;
  sum_to_zero_vector[K_R] a_R_raw;

  sum_to_zero_vector[K_L] a_L;
  
  // Dataset-specific window effects:
  // for each dataset d, the W effects sum to zero across windows.
  array[K_D] sum_to_zero_vector[K_W] a_DW_raw;
  
  // Scales for partial pooling
  real<lower=1e-4, upper=2> sigma_D;
  real<lower=1e-4, upper=2> sigma_E;
  real<lower=1e-4, upper=2> sigma_U;
  real<lower=1e-4, upper=2> sigma_G;
  real<lower=1e-4, upper=2> sigma_R;
  real<lower=1e-4, upper=2> sigma_DW;
  
  // Continuous covariate effects
  real beta_CS_user;
  real beta_CS_item;
  real beta_UA;
  
  // Observation noise
  real<lower=1e-4, upper=1> sigma_obs;
}

transformed parameters {
  vector[K_D] a_D = sigma_D * a_D_raw;
  vector[K_E] a_E = sigma_E * a_E_raw;
  vector[K_G] a_G = sigma_G * a_G_raw;
  vector[K_U] a_U = sigma_U * a_U_raw;
  vector[K_R] a_R = sigma_R * a_R_raw;
  
  array[K_D] vector[K_W] a_DW;
  
  for (d in 1:K_D) {
    a_DW[d] = sigma_DW * a_DW_raw[d];
  }
}

model {
  vector[N] mu;
  
  // Priors
  alpha ~ normal(mean_Y_logit, 1);
  
  a_D_raw ~ normal(0, 1);
  a_E_raw ~ normal(0, 1);
  a_G_raw ~ normal(0, 1);
  a_U_raw ~ normal(0, 1);
  a_R_raw ~ normal(0, 1);

  a_L ~ normal(0, 0.2);
  
  for (d in 1:K_D) {
    a_DW_raw[d] ~ normal(0, 1);
  }
  
  sigma_D  ~ normal(0, 0.5);
  sigma_E  ~ normal(0, 0.5);
  sigma_U  ~ normal(0, 0.5);
  sigma_G  ~ normal(0, 0.5);
  sigma_R  ~ normal(0, 0.5);
  sigma_DW ~ normal(0, 0.5);
  
  beta_CS_user ~ normal(0, 0.5);
  beta_CS_item ~ normal(0, 0.5);
  beta_UA      ~ normal(0, 0.5);
  
  sigma_obs ~ normal(0, 0.5);
  
  for (i in 1:N) {
    mu[i] =
      alpha
      + a_D[D_id[i]]
      + a_E[E_id[i]]
      + a_L[L_id[i]]
      + a_U[U_id[i]]
      + a_G[G_id[i]]
      + a_DW[D_id[i], W_id[i]]
      + a_R[R_id[i]]
      + beta_CS_user * CS_user[i]
      + beta_CS_item * CS_item[i]
      + beta_UA * UA[i];
  }
  
  Y_logit ~ normal(mu, sigma_obs * obs_scale);
}

generated quantities {
  vector[N] mu;
  vector[N] Y_hat;
  vector[N] log_lik;
  
  for (i in 1:N) {
    real sigma_i;
    
    mu[i] =
      alpha
      + a_D[D_id[i]]
      + a_E[E_id[i]]
      + a_L[L_id[i]]
      + a_U[U_id[i]]
      + a_G[G_id[i]]
      + a_DW[D_id[i], W_id[i]]
      + a_R[R_id[i]]
      + beta_CS_user * CS_user[i]
      + beta_CS_item * CS_item[i]
      + beta_UA * UA[i];
    
    sigma_i = sigma_obs * obs_scale[i];
    
    Y_hat[i] = inv_logit(mu[i]);
    log_lik[i] = normal_lpdf(Y_logit[i] | mu[i], sigma_i);
  }
}