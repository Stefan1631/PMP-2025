import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
def main():
    data = np.array([56, 60, 58,55,57,59,61,56,58,60])
    #definim modelul
    with pm.Model() as first_model:
        mu=pm.Normal("mu",mu=data.mean(),sigma=10)
        sigma=pm.HalfNormal("sigma",10)
        verosimilitate = pm.Normal("verosimilitate", mu=mu, sigma=sigma, observed=data)
        #facem inferenta
        idata = pm.sample(1000, random_seed=123, return_inferencedata=True)
        summary_first = az.summary(idata, var_names=["mu", "sigma"], hdi_prob=0.95)
    print("Rezumat posteriori:")
    print(f"Inferenta pe mu si sigma cu HDI 95%(intervalul este hdi_2.5% si hdi_97.5%):\n{summary_first}")

    print("estimarile frecventiste")
    #sample mean
    print(f"Sample mean: {np.mean(data):.2f}")
    #deviatia standard
    print(f"Deviatia standard: {np.std(data,ddof=1):.2f}")
    print(f"mu(58.013) este foarte apropiat de media frecventista(58),"
          f" iar sigma(2.321) este si el apropiat de deviatia standard frecventista(2).\n"
          f"Similaritatile provin din faptul ca distributia apriori aleasa este slaba,provenind din datele initiale,"
          f"ceea ce duce la o influenta foarte slaba a lor in calcularea verosimilitatii.")


    #acum facem un model mai puternic cu priori informativi
    with pm.Model() as new_model:
        mu=pm.Normal("mu",mu=50,sigma=1)
        sigma=pm.HalfNormal("sigma",10)
        verosimilitate = pm.Normal("verosimilitate", mu=mu, sigma=sigma, observed=data)
        idata_nou = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)
        summary_new = az.summary(idata_nou, var_names=["mu", "sigma"], hdi_prob=0.95)
    print(f"Rezumat posteriori:")
    print(summary_new)

    #facem graficele pt cele 2 estimari posterioare
    az.plot_posterior(idata, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Dist. posterioare cu datele slab informative", fontsize=14)
    plt.show()

    az.plot_posterior(idata_nou, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Dist. posterioare cu datele puternic informative", fontsize=14)
    plt.show()
    print("acum mu este mai mic decat valoarea frecventista,iar sigma este mai mare, intervalul pt HDI"
          "95% mutandu-se spre stanga.\nDeviatia standard creste datorita distributiei apriori ce influenteaza puternic distributia psoteriori")

if __name__ == "__main__":
    main()