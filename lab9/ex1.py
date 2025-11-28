import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
def main():
    #subpunctul a)

    Y=[0,5,10]
    THETA=[0.2,0.5]

    model_trace=dict()

    for y in Y:
        for theta in THETA:
            with pm.Model() as Model:
                n=pm.Poisson("n",mu=10)
                pm.Binomial("Y",n=n,p=theta,observed=y)
                trace=pm.sample(
                    draws=2000,
                    tune=2000,
                    chains=2,
                    random_seed=2025,
                    progressbar=False,
                )
            key=f"Y={y},Theta={theta}"
            model_trace[key]=trace

    fig, axes = plt.subplots(
        nrows=len(Y),
        ncols=len(THETA),
        figsize=(12, 10),
        constrained_layout=True
    )
    axes = axes.flatten()

    for i, (key, value) in enumerate(model_trace.items()):
        ax = axes[i]
        az.plot_posterior(
            data=value,
            var_names=['n'],
            ax=ax,
            hdi_prob=0.94,
            round_to=0,
            kind='hist',
            textsize=10
        )
        ax.set_title(f"pentru {key}")
        ax.set_xlabel("n")

    plt.suptitle("Distributiile Posterioare pentru n")
    plt.show()

    #subpunctul b)
    #Din grafice se observa ca cu cat Y creste, si range-ul pt n creste, deoarece Y este numarul
    #de clienti care au cumparat acel produs, iar n numarul total de clienti, deci n>=Y.
    #Cand theta creste (adica probabilitatea de a cumpara acel produs creste), modelul prezice ca
    #este nevoie de un numar mai mic de clienti n pentru acelasi Y (din grafic, range-ul pt n scade)


    #subpunctul c)

    predictive_posterior_dist=dict()

    for y in Y:
        for theta in THETA:
            key=f"Y={y},Theta={theta}"
            trace=model_trace[key]
            with pm.Model() as Model:
                n=pm.Poisson("n",mu=10)
                pm.Binomial("Y",n=n,p=theta,observed=y)
                predictive_posterior=pm.sample_posterior_predictive(
                    trace,
                    Model,
                    random_seed=2025
                )
            predictive_posterior_dist[key]=predictive_posterior


    fig, axes = plt.subplots(
        nrows=len(Y),
        ncols=len(THETA),
        figsize=(12, 10),
        constrained_layout=True
    )
    axes = axes.flatten()
    for i,(key, predictive) in enumerate(predictive_posterior_dist.items()):
        ax = axes[i]
        yStar = predictive.posterior_predictive['Y'].values.flatten()
        az.plot_dist(
            yStar,
            ax=ax,
            kind='hist',
            hist_kwargs={'bins': np.arange(0, np.max(yStar) + 1) - 0.5, 'rwidth': 0.9},
            rug=False
        )
        mean=np.mean(yStar)
        ax.axvline(mean, color='r', linestyle='--', label=f'Media: {mean:.2f}')

        ax.set_title(f"Distributia Predictiva Posterioara pentru Y*: {key}")
        ax.set_xlabel("Numar viitor de cumparatori")
        ax.legend()

    plt.suptitle(f"Distributiile Predictive Posterioare pentru Y*", fontsize=16, y=1.02)
    plt.show()

if __name__ == "__main__":
    main()