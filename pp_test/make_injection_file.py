import bilby
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(1010)

geocent_time = 1
prior = bilby.gw.prior.BBHPriorDict(filename="pp.prior")
prior["geocent_time"] = bilby.core.prior.Uniform(
    minimum=geocent_time - 0.1,
    maximum=geocent_time + 0.1,
    name="geocent_time",
)

n_injections = 100
samples = pd.DataFrame([prior.sample() for _ in range(n_injections)])
samples.to_csv("injections.csv", index_label="inj_id")
print(f"Saved {len(samples)} injections")
