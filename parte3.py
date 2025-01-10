import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy.stats import poisson
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
    'flat': True,
    'sigma8': 0.81,
    'ns': 0.96,
    'H0': 67.77,
    'Om0': 0.31,
    'Ob0': 0.049,
}
cosmology.addCosmology('euclidCosmo', params)
cosmology.setCosmology('euclidCosmo')

# Funzione di likelihood
def log_likelihood(theta, data, z, params, model='despali16'):
    sigma8, Om0 = theta
    if not (0.1 <= sigma8 <= 1.0 and 0.1 <= Om0 <= 1.0):
        return -np.inf

    params['sigma8'] = sigma8
    params['Om0'] = Om0
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

    if np.any(expected_counts <= 0) or np.any(np.isnan(expected_counts)):
        return -np.inf

    log_l = np.sum(poisson.logpmf(observed_counts, expected_counts))
    return log_l

# Inizializza i walker per MCMC
def initialize_walkers(initial, n_walkers, ndim, data, z, params):
    pos = []
    while len(pos) < n_walkers:
        p = initial + 0.05 * np.random.randn(ndim)
        if log_likelihood(p, data, z, params) > -np.inf:
            pos.append(p)
    return np.array(pos)

# Funzione per visualizzare i walker
def plot_walkers(sampler, labels):
    n_steps = sampler.chain.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = np.arange(n_steps)

    for i in range(2):  # Due parametri: sigma8 e Om0
        for walker in range(sampler.chain.shape[0]):  # Per ogni walker
            axes[i].plot(x, sampler.chain[walker, :, i], alpha=0.5)
        axes[i].set_ylabel(labels[i])

    axes[-1].set_xlabel("Steps")
    plt.tight_layout()
    plt.show()

# Esegui inferenza Bayesiana
def run_bayesian_inference(data, z, params, n_walkers=20, n_steps=1000, plot_walkers_flag=True):
    ndim = 2
    initial = [0.8, 0.3]
    pos = initialize_walkers(initial, n_walkers, ndim, data, z, params)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_likelihood, args=(data, z, params)
    )
    sampler.run_mcmc(pos, n_steps, progress=True)

    if plot_walkers_flag:
        labels = [r"$\sigma_8$", r"$\Omega_m$"]
        plot_walkers(sampler, labels)

    samples = sampler.get_chain(flat=True)
    labels = [r"$\sigma_8$", r"$\Omega_m$"]
    fig = corner.corner(samples, labels=labels, truths=[0.81, 0.31])
    plt.show()

    sigma8_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
    Om0_mcmc = np.percentile(samples[:, 1], [16, 50, 84])

    sigma8_final = sigma8_mcmc[1]
    Om0_final = Om0_mcmc[1]
    sigma8_uncertainty = (sigma8_mcmc[2] - sigma8_mcmc[1], sigma8_mcmc[1] - sigma8_mcmc[0])
    Om0_uncertainty = (Om0_mcmc[2] - Om0_mcmc[1], Om0_mcmc[1] - Om0_mcmc[0])

    print(f"Risultati per z = {z}:")
    print(f"  σ8 = {sigma8_final:.3f} (+{sigma8_uncertainty[0]:.3f}, -{sigma8_uncertainty[1]:.3f})")
    print(f"  Ωm = {Om0_final:.3f} (+{Om0_uncertainty[0]:.3f}, -{Om0_uncertainty[1]:.3f})")

    return samples

# Inferenza Bayesiana per ogni bin di redshift

# Liste per salvare i risultati
bin_centers = []
sigma8_values = []
sigma8_errors = []
Om0_values = []
Om0_errors = []

for z in bins:
    print(f"\n--- Inizio inferenza Bayesiana per z = {z:.2f} ---")
    samples = run_bayesian_inference(data, z, params, plot_walkers_flag=True)
    print(f"--- Fine inferenza Bayesiana per z = {z:.2f} ---")

    # Calcola i valori e gli errori
    sigma8_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
    Om0_mcmc = np.percentile(samples[:, 1], [16, 50, 84])
    sigma8_final = sigma8_mcmc[1]
    Om0_final = Om0_mcmc[1]
    sigma8_uncertainty = (sigma8_mcmc[2] - sigma8_mcmc[1], sigma8_mcmc[1] - sigma8_mcmc[0])
    Om0_uncertainty = (Om0_mcmc[2] - Om0_mcmc[1], Om0_mcmc[1] - Om0_mcmc[0])

    # Salva i risultati
    bin_centers.append(z)
    sigma8_values.append(sigma8_final)
    sigma8_errors.append(sigma8_uncertainty)
    Om0_values.append(Om0_final)
    Om0_errors.append(Om0_uncertainty)
    
    
    
    
    
    
def plot_parameters(bin_centers, sigma8_values, sigma8_errors, Om0_values, Om0_errors):
    """
    Crea un grafico con i valori di sigma8 e Omega_m per i vari bin di redshift.
    
    Parameters:
        bin_centers (list or array): Centri dei bin di redshift.
        sigma8_values (list or array): Valori di sigma8 ricavati.
        sigma8_errors (list of tuples): Errori su sigma8 (tuple di valori positivi e negativi).
        Om0_values (list or array): Valori di Omega_m ricavati.
        Om0_errors (list of tuples): Errori su Omega_m (tuple di valori positivi e negativi).
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Grafico per sigma8
    sigma8_yerr = np.array(sigma8_errors).T  # Trasposizione per separare errori superiori e inferiori
    ax[0].errorbar(
        bin_centers, sigma8_values, yerr=sigma8_yerr, fmt='o', label=r'$\sigma_8$', color='blue', capsize=5
    )
    ax[0].set_ylabel(r"$\sigma_8$")
    ax[0].grid(True)
    ax[0].legend()

    # Grafico per Omega_m
    Om0_yerr = np.array(Om0_errors).T  # Trasposizione per separare errori superiori e inferiori
    ax[1].errorbar(
        bin_centers, Om0_values, yerr=Om0_yerr, fmt='o', label=r'$\Omega_m$', color='red', capsize=5
    )
    ax[1].set_xlabel("Redshift (z)")
    ax[1].set_ylabel(r"$\Omega_m$")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    
# Grafico dei parametri
plot_parameters(bin_centers, sigma8_values, sigma8_errors, Om0_values, Om0_errors)


# Funzione per log-likelihood su tutto il dataset combinato
def log_likelihood_combined(theta, data, params, model='despali16'):
    sigma8, Om0 = theta
    if not (0.1 <= sigma8 <= 1.0 and 0.1 <= Om0 <= 1.0):
        return -np.inf

    # Aggiorna i parametri cosmologici
    params['sigma8'] = sigma8
    params['Om0'] = Om0
    cosmology.addCosmology('globalCosmo', params)
    cosmology.setCosmology('globalCosmo')

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

        if np.any(expected_counts <= 0) or np.any(np.isnan(expected_counts)):
            return -np.inf

        total_log_likelihood += np.sum(poisson.logpmf(observed_counts, expected_counts))

    return total_log_likelihood

# Funzione per inferenza Bayesiana sul dataset combinato
def run_global_inference(data, params, n_walkers=20, n_steps=1000, plot_walkers_flag=True):
    ndim = 2
    initial = [0.8, 0.3]
    pos = initialize_walkers(initial, n_walkers, ndim, data, bins[0], params)  # Primo bin per inizializzazione

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_likelihood_combined, args=(data, params)
    )
    sampler.run_mcmc(pos, n_steps, progress=True)

    if plot_walkers_flag:
        labels = [r"$\sigma_8$", r"$\Omega_m$"]
        plot_walkers(sampler, labels)

    samples = sampler.get_chain(flat=True)
    labels = [r"$\sigma_8$", r"$\Omega_m$"]
    fig = corner.corner(samples, labels=labels, truths=[0.81, 0.31])
    plt.show()

    sigma8_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
    Om0_mcmc = np.percentile(samples[:, 1], [16, 50, 84])

    sigma8_final = sigma8_mcmc[1]
    Om0_final = Om0_mcmc[1]
    sigma8_uncertainty = (sigma8_mcmc[2] - sigma8_mcmc[1], sigma8_mcmc[1] - sigma8_mcmc[0])
    Om0_uncertainty = (Om0_mcmc[2] - Om0_mcmc[1], Om0_mcmc[1] - Om0_mcmc[0])

    print("\n--- Risultati globali ---")
    print(f"  σ8 = {sigma8_final:.3f} (+{sigma8_uncertainty[0]:.3f}, -{sigma8_uncertainty[1]:.3f})")
    print(f"  Ωm = {Om0_final:.3f} (+{Om0_uncertainty[0]:.3f}, -{Om0_uncertainty[1]:.3f})")

    return samples

# Esegui inferenza globale
print("\n--- Inizio inferenza globale ---")
global_samples = run_global_inference(data, params, plot_walkers_flag=True)
print("--- Fine inferenza globale ---")

# Funzione per il grafico dei parametri cosmologici
def plot_parameters(bin_centers, sigma8_values, sigma8_errors, Om0_values, Om0_errors, sigma8_global=None, Om0_global=None):
    """
    Crea un grafico con i valori di sigma8 e Omega_m per i vari bin di redshift.
    
    Parameters:
        bin_centers (list or array): Centri dei bin di redshift.
        sigma8_values (list or array): Valori di sigma8 ricavati.
        sigma8_errors (list of tuples): Errori su sigma8 (tuple di valori positivi e negativi).
        Om0_values (list or array): Valori di Omega_m ricavati.
        Om0_errors (list of tuples): Errori su Omega_m (tuple di valori positivi e negativi).
        sigma8_global (float, optional): Valore globale di sigma8.
        Om0_global (float, optional): Valore globale di Omega_m.
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Grafico per sigma8
    sigma8_yerr = np.array(sigma8_errors).T  # Trasposizione per separare errori superiori e inferiori
    ax[0].errorbar(
        bin_centers, sigma8_values, yerr=sigma8_yerr, fmt='o', label=r'$\sigma_8$ (bins)', color='blue', capsize=5
    )
    if sigma8_global is not None:
        ax[0].axhline(y=sigma8_global, color='green', linestyle='--', label=r'$\sigma_8$ globale')
    ax[0].set_ylabel(r"$\sigma_8$")
    ax[0].grid(True)
    ax[0].legend()

    # Grafico per Omega_m
    Om0_yerr = np.array(Om0_errors).T  # Trasposizione per separare errori superiori e inferiori
    ax[1].errorbar(
        bin_centers, Om0_values, yerr=Om0_yerr, fmt='o', label=r'$\Omega_m$ (bins)', color='red', capsize=5
    )
    if Om0_global is not None:
        ax[1].axhline(y=Om0_global, color='green', linestyle='--', label=r'$\Omega_m$ globale')
    ax[1].set_xlabel("Redshift (z)")
    ax[1].set_ylabel(r"$\Omega_m$")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

# Grafico dei parametri, includendo i valori globali
plot_parameters(bin_centers, sigma8_values, sigma8_errors, Om0_values, Om0_errors, sigma8_global, Om0_global)