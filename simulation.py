import os
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma
from pathlib import Path
from cmdstanpy import CmdStanModel


# Directories
data_dir = Path("gamma_censoring/data")
model_dir = Path("gamma_censoring/models")
output_dir = Path("gamma_censoring/output")
for d in [data_dir, model_dir, output_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Step 1: Simulate Data
def simulate_censored_gamma(alpha, beta, N, T):
    scale = 1 / beta
    samples = np.random.gamma(shape=alpha, scale=scale, size=N)
    complete = samples[samples <= T]
    censored = samples[samples > T]
    return complete, np.full_like(censored, T), samples


# Step 2: Save JSON data for Stan
def prepare_stan_data(complete, censored, json_path):
    stan_data = {
        "N_obs": len(complete),
        "N_cens": len(censored),
        "obs": complete.tolist(),
        "cens": censored.tolist()
    }
    with open(json_path, "w") as f:
        json.dump(stan_data, f, indent=2)


# Step 3: Write Stan model
def write_stan_model(filepath):
    stan_code = """
    data {
      int<lower=0> N_obs;
      int<lower=0> N_cens;
      vector<lower=0>[N_obs] obs;
      vector<lower=0>[N_cens] cens;
    }
    parameters {
      real<lower=0> alpha;
      real<lower=0> beta;
    }
    model {
      target += gamma_lpdf(obs | alpha, beta);
      target += gamma_lccdf(cens | alpha, beta);
    }
    """
    with open(filepath, "w") as f:
        f.write(stan_code)


# Step 4: Fit with CmdStanPy
def fit_model(model_path, data_path, output_dir):
    model = CmdStanModel(stan_file=model_path)
    fit = model.sample(
        data=data_path,
        output_dir=output_dir,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        show_progress=True
    )
    return fit


def plot_stan_gamma_posterior(
    alphas,
    betas,
    true_alpha,
    true_beta,
    T,
    complete=None,
    output_path=None
):
    stan_means = alphas / betas
    stan_mean = np.mean(stan_means)
    stan_mean_std = np.std(stan_means)
    true_mean = true_alpha / true_beta

    x = np.linspace(0.001, 10, 500)
    pdfs = np.array([gamma.pdf(x, a=a, scale=1 / b) for a, b in zip(alphas, betas)])
    mean_pdf = np.mean(pdfs, axis=0)
    lower_pdf = np.percentile(pdfs, 2.5, axis=0)
    upper_pdf = np.percentile(pdfs, 97.5, axis=0)
    true_pdf = gamma.pdf(x, a=true_alpha, scale=1 / true_beta)

    plt.figure(figsize=(10, 6))

    # Plot histogram of observed (complete) wait times
    if complete is not None:
        plt.hist(
            complete,
            bins=30,
            density=True,
            alpha=0.4,
            color='gray',
            label="Observed (complete wait times)"
        )

    # Posterior uncertainty and densities
    plt.fill_between(x, lower_pdf, upper_pdf, color='blue', alpha=0.2, label="Stan PDF 95% CI")
    plt.plot(x, mean_pdf, label="Stan Mean PDF", linestyle='--', color='blue', linewidth=2)
    plt.plot(x, true_pdf, label="True Gamma PDF", color='green', linewidth=2)

    # Stan mean ± 1 SD vertical region
    plt.fill_betweenx(
        y=[0, plt.gca().get_ylim()[1]],
        x1=stan_mean - stan_mean_std,
        x2=stan_mean + stan_mean_std,
        color='blue',
        alpha=0.1,
        label="Stan mean ± 1 SD"
    )

    # Vertical reference lines
    plt.axvline(stan_mean, color='blue', linestyle='--',
                label=f"Stan mean ≈ {stan_mean:.2f} ± {stan_mean_std:.2f}", marker='^')
    plt.axvline(true_mean, color='green', linestyle='--', label=f"True mean = {true_mean:.2f}", marker='o')
    plt.axvline(T, color='red', linestyle='--', label=f"Censoring threshold T = {T}", marker='x')

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Stan Gamma Posterior: PDF and Wait Time Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_posterior_means(alphas, betas, true_alpha, true_beta, output_path=None):
    stan_means = alphas / betas
    true_mean = true_alpha / true_beta

    plt.figure(figsize=(8, 5))
    plt.hist(stan_means, bins=40, density=True, alpha=0.7, color='skyblue', label="Sampled Means")
    plt.axvline(true_mean, color='green', linestyle='--', linewidth=2, label=f"True Mean = {true_mean:.2f}")
    plt.xlabel("Mean of Gamma (alpha / beta)")
    plt.ylabel("Density")
    plt.title("Posterior Distribution of the Mean (Wait Time)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()


def main():
    np.random.seed(42)
    true_alpha, true_beta, N, T = 2.0, 1.5, 10000, 3.0

    complete, censored, all_samples = simulate_censored_gamma(true_alpha, true_beta, N, T)
    json_path = data_dir / "gamma_data.json"
    model_path = model_dir / "censored_gamma.stan"
    prepare_stan_data(complete, censored, json_path)
    write_stan_model(model_path)

    fit = fit_model(str(model_path), str(json_path), str(output_dir))

    plot_stan_gamma_posterior(
        alphas=fit.stan_variable("alpha"),
        betas=fit.stan_variable("beta"),
        true_alpha=true_alpha,
        true_beta=true_beta,
        T=T,
        complete=complete,
        output_path=Path("gamma_fit_plot.png")
    )
    plot_posterior_means(
        alphas=fit.stan_variable("alpha"),
        betas=fit.stan_variable("beta"),
        true_alpha=true_alpha,
        true_beta=true_beta,
        output_path=Path("gamma_mean_posterior.png")
    )


if __name__ == "__main__":
    main()
