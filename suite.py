import subprocess, time, os
import numpy as np
import sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials

params = {
    'latent_factors': [20, 50, 100, 200, 500],
    'eta': np.linspace(0.001, 0.03, 15),
    'reg': np.logspace(-4, -1, 15),
    'num_epochs': [50],
    'early_stopping': [0.0003]
}

for model in ['svd']:
    for lf in [str(lf) for lf in params['latent_factors']]:
        for eta in [str(eta) for eta in params['eta']]:
            for reg in [str(reg) for reg in params['reg']]:
                for epochs in [str(ep) for ep in params['num_epochs']]:
                    for early_stopping in [str(e) for e in params['early_stopping']]:
                        process = subprocess.run(['./run_' + model, lf, eta,
                            reg, epochs, early_stopping], stdout=subprocess.PIPE)
                        output = process.stdout.decode('utf-8').split('\n')
                        error_vals = {
                            'in_sample': float(output[1].split()[-1]),
                            'out_of_sample': float(output[2].split()[-1]),
                            'probe': float(output[3].split()[-1])
                        }
                        sheets.sheet.worksheet(model).append_row([
                            time.strftime('%c'), error_vals['in_sample'],
                            error_vals['out_of_sample'], error_vals['probe'],
                            lf, eta, reg, epochs, early_stopping
                        ])
