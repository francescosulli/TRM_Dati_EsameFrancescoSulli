import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy.stats import poisson, norm
import emcee
import corner

# Parte 1: Caricamento del dataset
dataset_path = "Euclid_ammassi.csv"
print("Tentativo di caricare il file:", dataset_path)

try:
    data = pd.read_csv(dataset_path)
    print("File caricato con successo!")
except FileNotFoundError:
    print(f"Errore: il file '{dataset_path}' non è stato trovato.")
    raise
except Exception as e:
    print(f"Errore durante il caricamento del file: {e}")
    raise

# Verifica che il dataset contenga le colonne richieste e filtra i dati non fisici
print("Verifica delle colonne richieste nel dataset...")
required_columns = {'mass', 'z', 'vol'}
if not required_columns.issubset(data.columns):
    raise ValueError(f"Il dataset deve contenere le colonne {required_columns}. Trovate: {data.columns.tolist()}")

# Stampa delle prime righe del file
print("\nPrime righe del dataset:")
print(data.head())

# Filtraggio dei dati
data = data[(data['mass'] > 0) & (data['z'] > 0) & (data['vol'] > 0)]
if data.empty:
    raise ValueError("Il dataset filtrato è vuoto dopo aver rimosso valori non fisici.")

# Identifica i bin di redshift univoci
bins = np.sort(data['z'].unique())
print("\nBin di redshift identificati:")
print(bins)

# Stampa delle prime righe per ogni bin di redshift
print("\nPrime righe per ciascun bin di redshift:")
for z in bins:
    subset = data[data['z'] == z]
    print(f"\nBin di redshift z = {z:.2f}:")
    print(subset.head())

# Impostazioni cosmologiche iniziali
params = {
    'flat': False,
    'sigma8': 0.81,
    'ns': 0.96,
    'H0': 67.77,
    'Om0': 0.31,
    'Ob0': 0.049,
    'Ode0': 0.7,
}
cosmology.addCosmology('euclidCosmo', params)
cosmology.setCosmology('euclidCosmo')

# Parte 2: Definizione della likelihood e funzioni necessarie
def log_likelihood(theta, data, z, params, model='despali16'):
    sigma8, Om0, Ode0 = theta
    if not (0.1 <= sigma8 <= 1.0 and 0.1 <= Om0 <= 1.0 and 0.1 <= Ode0 <= 1.0):
        return -np.inf

    params.update({'sigma8': sigma8, 'Om0': Om0, 'Ode0': Ode0})
    cosmology.addCosmology('bayesianCosmo', params)
    cosmology.setCosmology('bayesianCosmo')

    subset = data[data['z'] == z]
    if subset.empty:
        return -np.inf

    volumes = subset['vol'].values
    masses = subset['mass'].values
    m_min, m_max = masses.min(), masses.max()

    if len(masses) < 2 or m_min <= 0 or m_max <= 0:
        return -np.inf

    m_arr = np.logspace(np.log10(m_min), np.log10(m_max), 50)
    hmf = mass_function.massFunction(m_arr, z, mdef='vir', model=model, q_out='dndlnM')

    observed_counts, _ = np.histogram(masses, bins=m_arr)
    bin_widths = np.diff(np.log(m_arr))
    expected_counts = hmf[:-1] * bin_widths * volumes[0]

    # Normalizzazione
    expected_counts = np.maximum(expected_counts, 1e-5)

    log_l = np.sum(poisson.logpmf(observed_counts, expected_counts))

    # Prior gaussiano su Ode0
    ode0_mean, ode0_std = 0.7, 0.05
    log_l += norm.logpdf(Ode0, loc=ode0_mean, scale=ode0_std)

    return log_l

def initialize_walkers(initial, n_walkers, ndim, data, z, params):
    pos = []
    while len(pos) < n_walkers:
        p = initial + 0.05 * np.random.randn(ndim)
        if log_likelihood(p, data, z, params) > -np.inf:
            pos.append(p)
    return np.array(pos)

def plot_walkers(sampler, labels):
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    for i, label in enumerate(labels):
        axes[i].plot(sampler.get_chain()[:, :, i], "k", alpha=0.3)
        axes[i].set_ylabel(label)
    axes[-1].set_xlabel("Step number")
    plt.tight_layout()
    plt.show()

def run_bayesian_inference(data, z, params, n_walkers=20, n_steps=1000):
    ndim = 3  # Parametri: sigma8, Om0, Ode0
    initial = [0.8, 0.31, 0.7]
    pos = initialize_walkers(initial, n_walkers, ndim, data, z, params)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood, args=(data, z, params))
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Plot dei walker
    labels = [r"$\sigma_8$", r"$\Omega_m$", r"$\Omega_\Lambda$"]
    plot_walkers(sampler, labels)

    samples = sampler.get_chain(flat=True)
    corner.corner(samples, labels=labels, truths=[0.81, 0.31, 0.7])
    plt.show()

    # Analisi risultati
    sigma8_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
    Om0_mcmc = np.percentile(samples[:, 1], [16, 50, 84])
    Ode0_mcmc = np.percentile(samples[:, 2], [16, 50, 84])

    sigma8_final = sigma8_mcmc[1]
    Om0_final = Om0_mcmc[1]
    Ode0_final = Ode0_mcmc[1]

    Omega_k = 1 - Om0_final - Ode0_final

    print(f"Risultati per z = {z:.2f}:")
    print(f"  σ8 = {sigma8_final:.3f} (+{sigma8_mcmc[2] - sigma8_final:.3f}, -{sigma8_final - sigma8_mcmc[0]:.3f})")
    print(f"  Ωm = {Om0_final:.3f} (+{Om0_mcmc[2] - Om0_final:.3f}, -{Om0_final - Om0_mcmc[0]:.3f})")
    print(f"  ΩΛ = {Ode0_final:.3f} (+{Ode0_mcmc[2] - Ode0_final:.3f}, -{Ode0_final - Ode0_mcmc[0]:.3f})")
    print(f"  Ωk = {Omega_k:.3f}")

    return np.median(samples, axis=0)

# Parte 3: Esecuzione per ciascun bin di redshift
results = []
for z in bins:
    print(f"\n--- Inizio inferenza Bayesiana per z = {z:.2f} ---")
    result = run_bayesian_inference(data, z, params, n_walkers=20, n_steps=1000)
    sigma8_final, Om0_final, Ode0_final = result  # Ottieni i risultati finali
    results.append((z, sigma8_final, Om0_final, Ode0_final))  # Salva i risultati dei parametri
    print(f"--- Fine inferenza Bayesiana per z = {z:.2f} ---")

# Parte 4: Confronto finale tra i bin
results = np.array(results)
z_vals = results[:, 0]
sigma8_vals = results[:, 1]
Om0_vals = results[:, 2]
Ode0_vals = results[:, 3]

# Plot dei parametri
plt.errorbar(z_vals, sigma8_vals, fmt='o', label=r"$\sigma_8$")
plt.errorbar(z_vals, Om0_vals, fmt='o', label=r"$\Omega_m$")
plt.errorbar(z_vals, Ode0_vals, fmt='o', label=r"$\Omega_\Lambda$")
plt.xlabel("Redshift bin")
plt.ylabel("Parameter value")
plt.legend()
plt.title("Confronto tra i parametri nei diversi bin di redshift")
plt.show()

# Parte 5: Analisi globale

# Unione dei dati da tutti i bin di redshift per l'analisi globale
def log_likelihood_global(theta, data, params, model='despali16'):
    sigma8, Om0, Ode0 = theta
    if not (0.1 <= sigma8 <= 1.0 and 0.1 <= Om0 <= 1.0 and 0.1 <= Ode0 <= 1.0):
        return -np.inf

    params.update({'sigma8': sigma8, 'Om0': Om0, 'Ode0': Ode0})
    cosmology.addCosmology('bayesianCosmo', params)
    cosmology.setCosmology('bayesianCosmo')

    total_log_likelihood = 0

    for z in bins:
        subset = data[data['z'] == z]
        if subset.empty:
            continue

        volumes = subset['vol'].values
        masses = subset['mass'].values
        m_min, m_max = masses.min(), masses.max()

        if len(masses) < 2 or m_min <= 0 or m_max <= 0:
            continue

        m_arr = np.logspace(np.log10(m_min), np.log10(m_max), 50)
        hmf = mass_function.massFunction(m_arr, z, mdef='vir', model=model, q_out='dndlnM')

        observed_counts, _ = np.histogram(masses, bins=m_arr)
        bin_widths = np.diff(np.log(m_arr))
        expected_counts = hmf[:-1] * bin_widths * volumes[0]

        expected_counts = np.maximum(expected_counts, 1e-5)

        total_log_likelihood += np.sum(poisson.logpmf(observed_counts, expected_counts))

    # Prior gaussiano su Ode0
    ode0_mean, ode0_std = 0.7, 0.05
    total_log_likelihood += norm.logpdf(Ode0, loc=ode0_mean, scale=ode0_std)

    return total_log_likelihood

def run_global_bayesian_inference(data, params, n_walkers=20, n_steps=1000):
    ndim = 3  # Parametri: sigma8, Om0, Ode0
    initial = [0.8, 0.31, 0.7]
    pos = initialize_walkers(initial, n_walkers, ndim, data, bins[0], params)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood_global, args=(data, params))
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Plot dei walker
    labels = [r"$\sigma_8$", r"$\Omega_m$", r"$\Omega_\Lambda$"]
    plot_walkers(sampler, labels)

    samples = sampler.get_chain(flat=True)
    corner.corner(samples, labels=labels, truths=[0.81, 0.31, 0.7])
    plt.show()

    # Analisi risultati
    sigma8_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
    Om0_mcmc = np.percentile(samples[:, 1], [16, 50, 84])
    Ode0_mcmc = np.percentile(samples[:, 2], [16, 50, 84])

    sigma8_final = sigma8_mcmc[1]
    Om0_final = Om0_mcmc[1]
    Ode0_final = Ode0_mcmc[1]

    Omega_k = 1 - Om0_final - Ode0_final

    print(f"Risultati globali:")
    print(f"  σ8 = {sigma8_final:.3f} (+{sigma8_mcmc[2] - sigma8_final:.3f}, -{sigma8_final - sigma8_mcmc[0]:.3f})")
    print(f"  Ωm = {Om0_final:.3f} (+{Om0_mcmc[2] - Om0_final:.3f}, -{Om0_final - Om0_mcmc[0]:.3f})")
    print(f"  ΩΛ = {Ode0_final:.3f} (+{Ode0_mcmc[2] - Ode0_final:.3f}, -{Ode0_final - Ode0_mcmc[0]:.3f})")
    print(f"  Ωk = {Omega_k:.3f}")

    return np.median(samples, axis=0)

# Esecuzione dell'inferenza bayesiana globale
print("\n--- Inizio inferenza Bayesiana globale ---")
global_result = run_global_bayesian_inference(data, params, n_walkers=20, n_steps=1000)
sigma8_final, Om0_final, Ode0_final = global_result  # Ottieni i risultati finali
print(f"--- Fine inferenza Bayesiana globale ---")