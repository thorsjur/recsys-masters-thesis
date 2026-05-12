data {
  int<lower=1> N; // number of observations
  int<lower=1> S; // number of unique seeds
  int<lower=1> T; // number of unique windows
  
  array[N] int<lower=1, upper=2> cond_id; // 1 = condition A, 2 = condition B
  array[N] int<lower=1, upper=S> seed_id; // seed index per observation
  array[N] int<lower=1, upper=T> time_id; // window index per observation
  
  vector[N] y; // observed metric
}
transformed data {
  vector[T] alt_sign;
  real curr_sign = 1;
  
  for (t in 1 : T) {
    alt_sign[t] = curr_sign;
    curr_sign = -curr_sign;
  }
}
parameters {
  vector[2] alpha; // average performance by condition
  
  vector[S] b_seed_raw; // seed effects
  real<lower=1e-6> sigma_seed; // seed variability
  
  matrix[2, T] z_u;
  vector<lower=-0.8, upper=0.8>[2] rho; // AR(1) coefficient by condition
  vector<lower=1e-6>[2] sigma_u;
  
  real alternating_eff; // alternating window-to-window effect
  
  vector<lower=1e-6>[2] sigma_y;
}
transformed parameters {
  vector[S] b_seed;
  matrix[2, T] u;
  
  b_seed = sigma_seed * b_seed_raw;
  
  for (c in 1 : 2) {
    // Stationary initialization for AR(1) process
    u[c, 1] = z_u[c, 1] * sigma_u[c] / sqrt(1 - square(rho[c]));
    
    for (t in 2 : T) {
      u[c, t] = rho[c] * u[c, t - 1] + sigma_u[c] * z_u[c, t];
    }
  }
}
model {
  // Mean performance prior
  alpha ~ normal(0.4, 0.2);
  
  // Hierarchical seed effects
  b_seed_raw ~ normal(0, 1);
  sigma_seed ~ normal(0, 0.02);
  
  // AR(1) temporal process
  to_vector(z_u) ~ normal(0, 1);
  rho ~ normal(0, 0.3);
  sigma_u ~ normal(0, 0.02);
  
  // Alternating effect
  alternating_eff ~ normal(0, 0.05);
  
  // Residual noise
  sigma_y ~ normal(0, 0.02);
  
  for (n in 1 : N) {
    int c = cond_id[n];
    int t = time_id[n];
    
    real mu = alpha[c] + b_seed[seed_id[n]] + u[c, t]
              + alternating_eff * alt_sign[t];
    
    y[n] ~ normal(mu, sigma_y[c]);
  }
}
generated quantities {
  real mean_diff = alpha[2] - alpha[1];
  real alternating_eff_diff = 2 * alternating_eff;
  
  real general_stability_diff = sigma_y[2] - sigma_y[1];
  real temporal_stability_diff = sigma_u[2] - sigma_u[1];
  
  real general_stability_rel_gain = (sigma_y[1] - sigma_y[2]) / sigma_y[1];
  real temporal_stability_rel_gain = (sigma_u[1] - sigma_u[2]) / sigma_u[1];
  
  int<lower=0, upper=1> cond2_better = mean_diff > 0;
  int<lower=0, upper=1> cond2_more_generally_stable = general_stability_diff
                                                      < 0;
  int<lower=0, upper=1> cond2_more_temporally_stable = temporal_stability_diff
                                                       < 0;
}
