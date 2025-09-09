from MFDFA_functions import get_generalised_hurst, get_mf_spectrum
import numpy as np
import matplotlib.pyplot as plt
from MFDFA_new import MFDFA
import pandas as pd
from iaaft import surrogates
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class MFDFA_Analysis_Simulation:
    '''
    Class to perform MFDFA analysis on a type of returns / vol dataframe for simulated data with multiple paths. 
    '''

    def __init__(self, 
                datas: list, 
                burn_in: int, 
                s_min: int, 
                q_min: int, 
                q_max: int,
                m: int, 
                steps: int, 
                type: str, 
                models: list, 
                block: int, 
                plots: bool, 
                shuffle: bool, 
                compare_to_shuffle_ind: bool, 
                n_surrogates: int, 
                accuracy_surrogates: float, 
                prop_sims_for_surr: int) -> None: # doesnt return anything
        '''
        Parameters:
        - datas: List containing the raw returns/vol to analyse. Index order is same as models order. 
        - burn_in: Number of initial data points to discard.
        - s_min: Minimum scale for MFDFA.
        - q_min: Minimum order for generalized Hurst exponent.
        - q_max: Maximum order for generalized Hurst exponent.
        - m: Polynomial fitting for MFDFA.
        - steps: Number of steps for the segment array.
        - type: Type of data passed to the MFDFA analysis (squared, raw, volatility).
        - models: List of model names for the data.
        - block: Block size for block volatility calculation (if type is 'block_volatility').
        - plots: Boolean to indicate if plots should be generated.
        - shuffle: Boolean to indicate if data should be shuffled.
        - compare_to_shuffle_ind: Boolean to indicate if comparison to shuffled data should be done.
        - n_surrogates: Number of surrogate paths to generate for each path.
        - accuracy_surrogates: Accuracy of the surrogate paths.
        - prop_sims_for_surr: Proportion of simulations to use for surrogate generation.
        '''

        self.datas = datas
        self.burn_in = burn_in
        self.s_min = s_min
        self.q_min = q_min
        self.q_max = q_max
        self.m = m
        self.steps = steps
        self.type = type
        self.models = models
        self.block = block
        self.plots = plots
        self.shuffle = shuffle
        self.compare_to_shuffle_ind = compare_to_shuffle_ind
        self.n_surrogates = n_surrogates
        self.accuracy_surrogates = accuracy_surrogates
        self.prop_sims_for_surr = prop_sims_for_surr

        self.datas_clean = None
        self.segments = None
        self.qs = None
        self.s_max = None
        self.npaths = datas[0].shape[1]
        self.datas_clean_vol = []
        self.hursts = []
        self.alphas = []
        self.f_alphas = []
        self.taos = []
        self.datas_clean_shuffled = None
        self.hursts_shuffled = []
        self.alphas_shuffled = []
        self.f_alphas_shuffled = []
        self.taos_shuffled = []
        self.datas_clean_surrogates = []
        self.hursts_surrogates = []
        self.alphas_surrogates = []
        self.f_alphas_surrogates = []
        self.taos_surrogates = []
        self.p_values = []

    def get_block_vol(self, returns):

            N = len(returns)
            n_obs = N // self.block

            vols = np.zeros((n_obs))

            for i in range(n_obs):
                vols[i] = np.std(returns[i * self.block : (i+1) * self.block])

            return vols

    def clean_data(self):

        self.datas_clean = [data[self.burn_in:] for data in self.datas] # do not modify datas in place

        if self.type == 'raw_log_returns' or self.type == 'underlying_volatility' or self.type == 'underlying_variance' or self.type == 'raw_log_returns_MSM' or self.type == 'raw_log_returns_MSMa':
            pass
        elif self.type == 'squared_returns' or self.type == 'squared_returns_MSM' or self.type == 'squared_returns_MSMa':
            self.datas_clean = [data ** 2 for data in self.datas_clean]
        elif self.type == 'absolute_returns' or self.type == 'absolute_returns_MSM' or self.type == 'absolute_returns_MSMa':
            self.datas_clean = [np.abs(data) for data in self.datas_clean]
        elif self.type == 'block_volatility':
            if self.block is None:
                raise ValueError("Block size must be specified for block_volatility type")
            
            self.datas_clean_vol = []  
            
            for i in range(len(self.models)):
                model_vol_data = []
                for j in range(self.npaths):
                    vol_path = self.get_block_vol(self.datas_clean[i][:, j])
                    model_vol_data.append(vol_path)
                
                model_vol_array = np.array(model_vol_data).T
                self.datas_clean_vol.append(model_vol_array)
            
    
    def get_f_and_h(self, X):
        '''
        Does MF-DFA with m for X a single time series path.
        '''

        F_qs = list()

        for q in self.qs:

            _, F = MFDFA(X, lag = self.segments, order = self.m, q=q)
            F_qs.append(F)

        generalised_H = np.zeros(len(self.qs))

        for i in range(len(F_qs)): 

            generalised_H[i] = get_generalised_hurst(F_q = F_qs[i], ss = self.segments)

        tao_q, alpha_q, f_alpha = get_mf_spectrum(generalised_H, self.qs)

        return generalised_H, alpha_q, f_alpha, tao_q
    

    def do_analysis(self):

        self.clean_data()

        if self.type == "block_volatility":
            s_max = self.datas_clean_vol[0].shape[0] // 4
        else:
            s_max = self.datas_clean[0].shape[0] // 4 

        self.segments = np.logspace(np.log10(self.s_min), np.log10(s_max), num=self.steps)
        self.segments = np.unique(np.round(self.segments)).astype(int)

        self.qs = np.arange(self.q_min, self.q_max + 1, 1)

        if self.type == "block_volatility":
            df = self.datas_clean_vol
        else:
            df = self.datas_clean

        for i in range(len(self.models)):
            model_data = df[i]
            model_hursts = []
            model_alphas = []
            model_f_alphas = []
            model_taos = []

            for j in range(self.npaths):

                if j % 100 == 0:
                    print(f"Processing model {self.models[i]}, path {j+1}/{self.npaths}")

                path_data = model_data[:, j]
                generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(path_data)

                model_hursts.append(generalised_H)
                model_alphas.append(alpha_q)
                model_f_alphas.append(f_alpha)
                model_taos.append(tao_q)

            self.hursts.append(np.array(model_hursts))
            self.alphas.append(np.array(model_alphas))
            self.f_alphas.append(np.array(model_f_alphas))
            self.taos.append(np.array(model_taos))

        return self.hursts, self.alphas, self.f_alphas, self.taos, self.segments, self.qs
    
    def do_analysis_shuffled(self):

        self.clean_data()

        if self.type == "block_volatility":
            s_max = self.datas_clean_vol[0].shape[0] // 4
        else:
            s_max = self.datas_clean[0].shape[0] // 4 

        self.segments = np.logspace(np.log10(self.s_min), np.log10(s_max), num=self.steps)
        self.segments = np.unique(np.round(self.segments)).astype(int)

        self.qs = np.arange(self.q_min, self.q_max + 1, 1)

        if self.type == "block_volatility":
            df = self.datas_clean_vol
        else:
            df = self.datas_clean

        for i in range(len(self.models)):
            curr_data = df[i]
            curr_shuff = np.zeros_like(curr_data)
            for j in range(self.npaths):
                perm_rows = np.random.permutation(curr_data.shape[0])
                curr_shuff[:, j] = curr_data[perm_rows, j]

            df[i] = curr_shuff

        for i in range(len(self.models)):
            model_data = df[i]
            model_hursts = []
            model_alphas = []
            model_f_alphas = []
            model_taos = []

            for j in range(self.npaths):

                if j % 100 == 0:
                    print(f"Processing model {self.models[i]}, path {j+1}/{self.npaths} (Shuffled)")

                path_data = model_data[:, j]
                generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(path_data)

                model_hursts.append(generalised_H)
                model_alphas.append(alpha_q)
                model_f_alphas.append(f_alpha)
                model_taos.append(tao_q)

            self.hursts_shuffled.append(np.array(model_hursts))
            self.alphas_shuffled.append(np.array(model_alphas))
            self.f_alphas_shuffled.append(np.array(model_f_alphas))
            self.taos_shuffled.append(np.array(model_taos))

        return self.hursts_shuffled, self.alphas_shuffled, self.f_alphas_shuffled, self.taos_shuffled, self.segments, self.qs
    
    def get_plots(self):
        '''
        Plot MFDFA results with error bars on average MF spectrum. These error bars
        represent the standard error of the mean f_alpha metric. This is computed with:
        SE(\bar{f_alpha}) = \frac{\sigma_{f_alpha}}{\sqrt{n}})
        '''

        colors = ['black', 'blue','green', 'red', 'orange', 'purple']
        markers = ['o', 's', '^', 'd', 'v', '<']

        if self.plots:

            if self.shuffle:
                self.do_analysis_shuffled()
                H = self.hursts_shuffled
                A = self.alphas_shuffled
                F = self.f_alphas_shuffled
                T = self.taos_shuffled
            else:
                self.do_analysis()
                H = self.hursts
                A = self.alphas
                F = self.f_alphas
                T = self.taos

            plt.style.use('seaborn-v0_8-dark')
            plt.figure(figsize=(18, 12))

            # average MF spectrum with error bars
            plt.subplot(2, 3, 1)
            q0_idx = np.where(self.qs == 0)[0][0]
            for i in range(len(self.models)):
                avg_f_alpha = np.mean(F[i], axis=0)
                avg_alpha = np.mean(A[i], axis=0)
                avg_alpha_0 = avg_alpha[q0_idx]
                err_f_alpha = np.std(F[i], axis=0) / np.sqrt(self.npaths)
                plt.errorbar(avg_alpha, avg_f_alpha, yerr=err_f_alpha, capsize=10, alpha = 0.5, 
                             color = colors[i], marker = markers[i], label=self.models[i], fmt='-')
                plt.axvline(x=avg_alpha_0, color=colors[i], linestyle='--', alpha=0.5)
            plt.xlabel(r'Singularity Strength, $\alpha$', fontsize=14)
            plt.ylabel(r'Multifractal Spectrum, $f(\alpha)$', fontsize=14)
            plt.title('Average Multifractal Spectrum with SE bars', fontsize=16)
            plt.legend(fontsize='medium')
            plt.grid(True, alpha=0.3)

            # average Hurst exponent with error bars
            plt.subplot(2, 3, 2)
            for i in range(len(self.models)):
                avg_hurst = np.mean(H[i], axis=0)
                err_hurst = np.std(H[i], axis=0) / np.sqrt(self.npaths)
                plt.errorbar(self.qs, avg_hurst, yerr=err_hurst, capsize=10, alpha=0.5,
                             color=colors[i], marker=markers[i], label=self.models[i], fmt='-')
            plt.xlabel(r'Order, $q$', fontsize=14)
            plt.ylabel(r'Hurst Exponent, $H(q)$', fontsize=14)
            plt.title('Average Hurst Exponent with SE bars', fontsize=16)
            plt.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='H=0.5')
            plt.legend(fontsize='medium')
            plt.grid(True, alpha=0.3)

            # average tao with error bars
            plt.subplot(2, 3, 3)
            for i in range(len(self.models)):
                avg_tao = np.mean(T[i], axis=0)
                err_tao = np.std(T[i], axis=0) / np.sqrt(self.npaths)
                plt.errorbar(self.qs, avg_tao, yerr=err_tao, capsize=10, alpha=0.5,
                             color=colors[i], marker=markers[i], label=self.models[i], fmt='-')
            plt.xlabel(r'Order, $q$', fontsize=14)
            plt.ylabel(r'$\tau(q)$', fontsize=14)
            plt.title(r'Average $\tau(q)$ with SE bars', fontsize=16)
            plt.legend(fontsize='medium')
            plt.grid(True, alpha=0.3)

            # width of the spectra
            plt.subplot(2, 3, 4)
            for i in range(len(self.models)):
                A_finite = A[i][np.isfinite(A[i])]
                if len(A_finite) > 0:
                    A_reshaped = A[i].copy()
                    A_reshaped[~np.isfinite(A_reshaped)] = np.nan
                    widths = np.nanmax(A_reshaped, axis=1) - np.nanmin(A_reshaped, axis=1)
                    widths = widths[np.isfinite(widths)]
                else:
                    widths = np.array([])
                
                if len(widths) > 0:
                    plt.boxplot(widths, positions=[i], tick_labels = [self.models[i]], widths=0.4)
            plt.ylabel(r'$\alpha_{\max} - \alpha_{\min}$', fontsize=14)
            plt.title('Width of Multifractal Spectrum', fontsize=16)
            plt.tick_params(axis='x', labelsize=14)
            plt.grid(True, alpha=0.3)

            # skewness of the spectra
            plt.subplot(2, 3, 5)
            q0_idx = np.where(self.qs == 0)[0][0]

            for j in range(len(self.models)):
                skew = []
                for i in range(self.npaths):
                    alpha_0 = A[j][i, q0_idx]
                    alpha_finite = A[j][i, :][np.isfinite(A[j][i, :])]
                    if len(alpha_finite) > 0:
                        alpha_max = np.max(alpha_finite)
                        alpha_min = np.min(alpha_finite)
                        if np.isfinite(alpha_0) and alpha_0 != alpha_min:
                            skew.append((alpha_max - alpha_0) / (alpha_0 - alpha_min))
                if len(skew) > 0:
                    plt.boxplot(skew, positions=[j], tick_labels=[self.models[j]], widths=0.4)

            plt.ylabel('Skewness', fontsize=14)
            plt.title('Skewness of Multifractal Spectrum', fontsize=16)
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Un-skewed')
            plt.tick_params(axis='x', labelsize=14)
            plt.grid(True, alpha=0.3)

            # H(2)
            plt.subplot(2, 3, 6)
            q2_idx = np.where(self.qs == 2)[0][0]

            for i in range(len(self.models)):
                h2 = H[i][:, q2_idx]
                plt.boxplot(h2, positions=[i], tick_labels=[self.models[i]], widths=0.4)
            plt.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='H=0.5')
            plt.ylabel(r'H(2)', fontsize=14)
            plt.title('Distribution of Hurst Exponents', fontsize=16)
            plt.tick_params(axis='x', labelsize=14)
            plt.legend(fontsize='large')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.suptitle(f'MFDFA Simulation Analysis - {self.type} - m={self.m} - {"Shuffled" if self.shuffle else "Original"}', fontsize=18, fontweight='bold')
            plt.subplots_adjust(top=0.91) 
            if self.shuffle:
                plt.savefig(f'/Users/alexvillamartin/Documents/MSc Diss/plots_new/MFDFA_Analysis_Simulation_{self.type}_{self.m}_shuffle.png')
            else:
                plt.savefig(f'/Users/alexvillamartin/Documents/MSc Diss/plots_new/MFDFA_Analysis_Simulation_{self.type}_{self.m}.png')
            plt.show()


            # MF spectrum with error bars
            plt.figure(figsize=(10, 6))
            q0_idx = np.where(self.qs == 0)[0][0]

            q0_idx = np.where(self.qs == 0)[0][0]
            colors_2 = ['black', 'blue', 'blue', 'green']
            for i in range(len(self.models)):

                if i==2:

                    avg_f_alpha = np.mean(F[i], axis=0)
                    avg_alpha = np.mean(A[i], axis=0)
                    avg_alpha_0 = avg_alpha[q0_idx]
                    err_f_alpha = np.std(F[i], axis=0) / np.sqrt(self.npaths)
                    plt.errorbar(avg_alpha, avg_f_alpha, yerr=err_f_alpha, capsize=10, alpha = 0.25, 
                                color = 'blue', marker = markers[i], label=self.models[i], fmt='-')
                    plt.axvline(x=avg_alpha_0, color='blue', linestyle='--', alpha=0.25)

                else:
                    avg_f_alpha = np.mean(F[i], axis=0)
                    avg_alpha = np.mean(A[i], axis=0)
                    avg_alpha_0 = avg_alpha[q0_idx]
                    err_f_alpha = np.std(F[i], axis=0) / np.sqrt(self.npaths)
                    plt.errorbar(avg_alpha, avg_f_alpha, yerr=err_f_alpha, capsize=10, alpha = 0.5, 
                                color = colors_2[i], marker = markers[i], label=self.models[i], fmt='-')
                    plt.axvline(x=avg_alpha_0, color=colors_2[i], linestyle='--', alpha=0.5)

            plt.xlabel(r'Singularity Strength, $\alpha$', fontsize=14)
            plt.ylabel(r'Multifractal Spectrum, $f(\alpha)$', fontsize=14)
            plt.title('Multifractal Spectrum of Absolute Returns for Volatility Models', fontsize=18)
            plt.legend(fontsize='medium', loc='upper right')
            plt.grid(True, alpha=0.7)


    def get_widths(self,alphas):
        
        alphas_finite = alphas.copy()
        alphas_finite[~np.isfinite(alphas_finite)] = np.nan
        widths = np.nanmax(alphas_finite, axis=1) - np.nanmin(alphas_finite, axis=1)

        return widths

    def compare_to_shuffle(self):
        '''
        Compare MFDFA results to shuffled data.
        '''

        if self.shuffle and self.compare_to_shuffle_ind:
            self.do_analysis()
            A = self.alphas

            self.do_analysis_shuffled()
            A_shuffled = self.alphas_shuffled

        widths_original = []
        widths_shuffled = []
        
        for j in range(len(self.models)):
                widths_original.append(self.get_widths(A[j]))
                widths_shuffled.append(self.get_widths(A_shuffled[j]))

        plt.figure(figsize=(8, 6))

        positions_shuffled = np.arange(1, len(self.models)*2, 2) - 0.2
        positions_normal = np.arange(1, len(self.models)*2, 2) + 0.2

        bp1 = plt.boxplot(widths_shuffled, positions=positions_shuffled, widths=0.3, 
                        patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.7))
        bp2 = plt.boxplot(widths_original, positions=positions_normal, widths=0.3, 
                        patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))

        plt.xticks(np.arange(1, len(self.models)*2, 2), self.models)
        plt.ylabel('Spectrum Width', fontsize=12)
        plt.title(f'Comparison of Spectrum Widths (Shuffled vs. Original) - {self.type} - m={self.m}', fontsize=14)
        plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Shuffled', 'Original'], loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'/Users/alexvillamartin/Documents/MSc Diss/plots_new/MFDFA_Analysis_Simulation_{self.type}_{self.m}_Shuffled_Width_Comparison.png')
        plt.tight_layout()
        plt.show()

    def do_analysis_surrogates(self):

        self.clean_data()
        scale = self.prop_sims_for_surr
        n_surrogate_paths = self.npaths // scale

        if self.type == "block_volatility":
            s_max = self.datas_clean_vol[0].shape[0] // 4
        else:
            s_max = self.datas_clean[0].shape[0] // 4 

        self.segments = np.logspace(np.log10(self.s_min), np.log10(s_max), num=self.steps)
        self.segments = np.unique(np.round(self.segments)).astype(int)

        self.qs = np.arange(self.q_min, self.q_max + 1, 1)

        if self.type == "block_volatility":
            df = self.datas_clean_vol
        else:
            df = self.datas_clean

        for i in range(len(self.models)):
            curr_data = df[i]
            curr_surrogates = np.zeros((n_surrogate_paths, curr_data.shape[0], self.n_surrogates))

            for j in range(n_surrogate_paths):
                if j % 4 == 0:
                    print(f"Generating surrogates for model {self.models[i]}, path {j+1}/{n_surrogate_paths}")

                curr_data_path = curr_data[:, j*scale]
                curr_surrogates[j] = surrogates(curr_data_path, ns=self.n_surrogates, tol_pc=self.accuracy_surrogates, verbose=False).T

            self.datas_clean_surrogates.append(curr_surrogates)

        for i in range(len(self.models)):
            model_data = self.datas_clean_surrogates[i]
            model_hursts = []
            model_alphas = []
            model_f_alphas = []
            model_taos = []

            for j in range(n_surrogate_paths):
                if j % 4 == 0:
                    print(f"Processing surrogates for model {self.models[i]}, path {j+1}/{n_surrogate_paths}")

                path_data = model_data[j]
                path_hursts = []
                path_alphas = []
                path_f_alphas = []
                path_taos = []

                for k in range(self.n_surrogates):
                    surrogate_data = path_data[:, k]
                    generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(surrogate_data)

                    path_hursts.append(generalised_H)
                    path_alphas.append(alpha_q)
                    path_f_alphas.append(f_alpha)
                    path_taos.append(tao_q)

                model_hursts.append(path_hursts)
                model_alphas.append(path_alphas)
                model_f_alphas.append(path_f_alphas)
                model_taos.append(path_taos)

            self.hursts_surrogates.append(np.array(model_hursts))
            self.alphas_surrogates.append(np.array(model_alphas))
            self.f_alphas_surrogates.append(np.array(model_f_alphas))
            self.taos_surrogates.append(np.array(model_taos))

        return self.hursts_surrogates, self.alphas_surrogates, self.f_alphas_surrogates, self.taos_surrogates, self.segments, self.qs
    
    def do_analysis_selected_paths(self):

        self.clean_data()
        scale = self.prop_sims_for_surr
        n_surrogate_paths = self.npaths // scale

        if self.type == "block_volatility":
            s_max = self.datas_clean_vol[0].shape[0] // 4
        else:
            s_max = self.datas_clean[0].shape[0] // 4 

        self.segments = np.logspace(np.log10(self.s_min), np.log10(s_max), num=self.steps)
        self.segments = np.unique(np.round(self.segments)).astype(int)

        self.qs = np.arange(self.q_min, self.q_max + 1, 1)

        if self.type == "block_volatility":
            df = self.datas_clean_vol
        else:
            df = self.datas_clean

        for i in range(len(self.models)):
            model_data = df[i]
            model_hursts = []
            model_alphas = []
            model_f_alphas = []
            model_taos = []

            for j in range(n_surrogate_paths):

                if j % 4 == 0:
                    print(f"Processing model {self.models[i]}, path {j+1}/{n_surrogate_paths}")

                path_data = model_data[:, j*scale]
                generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(path_data)

                model_hursts.append(generalised_H)
                model_alphas.append(alpha_q)
                model_f_alphas.append(f_alpha)
                model_taos.append(tao_q)

            self.hursts.append(np.array(model_hursts))
            self.alphas.append(np.array(model_alphas))
            self.f_alphas.append(np.array(model_f_alphas))
            self.taos.append(np.array(model_taos))

        return self.hursts, self.alphas, self.f_alphas, self.taos, self.segments, self.qs

    def surrogate_tests(self):
        '''
        Perform surrogate tests to assess the significance of the MF present. This breaks nonlinear correlations and 
        preserves linear and fat tails, both of which can contribute to MF so this ensures we test to see if there is a true
        difference attributed to multifractlity. Do this on simulated first as we know what results should be. 

        Null is that the MF spectrum width of the original data is not significantly different from the surrogate data, hence no significant
        MF present.
        '''

        self.do_analysis_surrogates()
        self.do_analysis_selected_paths() # for comparison so we only have n_surrogate_paths (ie 10 rather than 500)

        scale = self.prop_sims_for_surr
        n_surrogate_paths = self.npaths // scale

        for j in range(len(self.models)):
            alphas_model = self.alphas[j]
            alphas_surrogates_model = self.alphas_surrogates[j]

            widths_original_per_path = self.get_widths(alphas_model)
            p_vals_model = []

            for i in range(n_surrogate_paths):
                
                widths_surrogates_curr_path = self.get_widths(alphas_surrogates_model[i, :, :])
                p_val = ((widths_surrogates_curr_path >= widths_original_per_path[i]).sum() + 1) / (self.n_surrogates + 1)
                p_vals_model.append(p_val)

            self.p_values.append(np.array(p_vals_model))

        return self.p_values
    


# ADD HURSTS ETC AND SURROGATE DF TO BE OUTPUTED WITH P VALUES ABOVE TOO
#####################################################

class MFDFA_Analysis_Empirical:
    '''
    Class to perform MFDFA analysis on a type of returns / vol dataframe for empirical data with daily / S5 data
    and selecting relevant trading hours and detrending. Unlike in simulation one we do not compute block vol yet and instead
    compute daily vol from 5 second data and compare this to daily squared returns. Add block vol later if needed.

    Note: Daily intraday vol of secondly data is compared to squared daily returns of daily data in the spectrum plots.  
    '''

    def __init__(self, 
                datas: list, 
                fraction_of_s5: float, 
                s_min_d: int, 
                s_min_s5: int,
                q_min: int, 
                q_max: int,
                m: int, 
                steps: int, 
                type: str, 
                models: list, 
                plots: bool, 
                end_trading_hr_euro: int,
                end_trading_hr_sgd: int, 
                start_trading_hr_euro: int,
                start_trading_hr_sgd: int, 
                unit_test: bool, 
                shuffle: bool, 
                compare_to_shuffle_ind: bool, 
                n_surrogates: int, 
                accuracy_surrogates: float) -> None: # doesnt return anything
        '''
        Parameters:
        - datas: List containing the raw returns/vol to analyse. Index order is same as models order. 
        - fraction_of_s5: Fraction of 5 second data to use for analysis (e.g. 10 means every 10th sample).
        - s_min_d: Minimum scale for daily data.
        - s_min_s5: Minimum scale for 5 second data.
        - q_min: Minimum order for generalized Hurst exponent.
        - q_max: Maximum order for generalized Hurst exponent.
        - m: Polynomial fitting for MFDFA.
        - steps: Number of steps for the segment array.
        - type: Type of data passed to the MFDFA analysis (squared, raw, volatility).
        - models: List of model names for the data.
        - plots: Boolean to indicate if plots should be generated.
        - end_trading_hr_euro: End trading hour for Euro data.
        - end_trading_hr_sgd: End trading hour for SGD data.
        - start_trading_hr_euro: Start trading hour for Euro data.
        - start_trading_hr_sgd: Start trading hour for SGD data.
        - unit_test: Boolean to indicate if unit tests should be run.
        - shuffle: Boolean to indicate if data should be shuffled for analysis.
        - compare_to_shuffled: Boolean to indicate if data should be compared to shuffled version for analysis.
        - n_surrogates: Number of surrogates to generate for surrogate tests.
        - accuracy_surrogates: Accuracy for surrogate tests.
        '''

        self.datas = datas
        self.fraction_of_s5 = fraction_of_s5
        self.s_min_d = s_min_d
        self.s_min_s5 = s_min_s5
        self.q_min = q_min
        self.q_max = q_max
        self.m = m
        self.steps = steps
        self.type = type
        self.models = models
        self.plots = plots
        self.end_trading_hr_euro = end_trading_hr_euro
        self.end_trading_hr_sgd = end_trading_hr_sgd
        self.start_trading_hr_euro = start_trading_hr_euro
        self.start_trading_hr_sgd = start_trading_hr_sgd
        self.unit_test = unit_test
        self.shuffle = shuffle
        self.compare_to_shuffle_ind = compare_to_shuffle_ind
        self.n_surrogates = n_surrogates
        self.accuracy_surrogates = accuracy_surrogates

        self.datas_clean = None
        self.segments_d = None
        self.segments_s5 = None
        self.qs = None
        self.s_max = None
        self.hursts = []
        self.alphas = []
        self.f_alphas = []
        self.taos = []
        self.datas_clean_shuffled = None
        self.hursts_shuffled = []
        self.alphas_shuffled = []
        self.f_alphas_shuffled = []
        self.taos_shuffled = []
        self.datas_clean_surrogates = []
        self.hursts_surrogates = []
        self.alphas_surrogates = []
        self.f_alphas_surrogates = []
        self.taos_surrogates = []
        self.p_values = []

    def filter_trading_hours(self, df, start_hour, end_hour):
        '''
        Make sure we select the trading hours we are interested in.
        '''

        df_copy = df.copy()
        
        mask = (df_copy.index.time >= pd.Timestamp(f'{start_hour:02d}:00:05').time()) & (df_copy.index.time <= pd.Timestamp(f'{end_hour:02d}:00:00').time())
        
        return df_copy[mask]

    def remove_first_day_and_last(self, df):
        '''
        Remove first and last day incase they dont start at the same time as the rest of the data.
        '''

        df_copy = df.copy()

        first_date = df_copy.index.date[0] 
        last_date = df_copy.index.date[-1]

        mask = (df_copy.index.date != first_date) & (df_copy.index.date != last_date)

        return df_copy[mask]
    
    def check_complete_5s_intervals(self, df, start_hour, end_hour):
        '''
        This function overlays a full 5 second index onto data and ffils missing values to ensure all intervals in 
        specified hours are there. 
        '''

        df_copy = df.copy()
        
        start_date = df_copy.index.min().floor('D')
        end_date = df_copy.index.max().ceil('D')
        full_index = pd.date_range(start=start_date, end=end_date, freq='5s')
        
        trading_mask = (full_index.time >= pd.Timestamp(f'{start_hour:02d}:00:05').time()) & (full_index.time <= pd.Timestamp(f'{end_hour:02d}:00:00').time())
        weekday_mask = full_index.dayofweek < 5 

        trading_index = full_index[weekday_mask & trading_mask]
        
        df_complete = df_copy.reindex(trading_index).ffill().bfill() # in case of gaps at start
        
        return df_complete
    
    def unit_test_trading_hours(self):
        '''
        Unit test to check if trading hours are correctly filtered.
        '''
        euro_s5_clean = self.datas_clean[2]
        sgd_s5_clean = self.datas_clean[3]

        print("Euro S5 data info:")
        print(f"Start: {euro_s5_clean.index.min()}")
        print(f"End: {euro_s5_clean.index.max()}")
        print(f"Total observations: {len(euro_s5_clean)}")
        print(f"Expected 5 second intervals per day (9 hours): {9 * 60 * 60 / 5}")
        print(f"Number of trading days: {len(pd.Series(euro_s5_clean.index.date).unique())}")
        print(f"Expected total observations: {len(pd.Series(euro_s5_clean.index.date).unique()) * 9 * 60 * 60 / 5}")

        print("\nSGD S5 data info:")
        print(f"Start: {sgd_s5_clean.index.min()}")
        print(f"End: {sgd_s5_clean.index.max()}")
        print(f"Total observations: {len(sgd_s5_clean)}")
        print(f"Number of trading days: {len(pd.Series(sgd_s5_clean.index.date).unique())}")

        print(f"\nEuro missing values: {euro_s5_clean.isnull().sum().sum()}")
        print(f"SGD missing values: {sgd_s5_clean.isnull().sum().sum()}")

    def get_sum_squared_intraday_returns(self, df):
        '''
        RV = \sum_{i=1}^{n} r_i^2 where n is number of returns in the day.
        '''
        idx = pd.Series(df.index.date).unique()
        n_days = pd.Series(df.index.date).nunique()
        K = len(df) // n_days  # number of returns per day
        
        vol = np.sum(df.values.reshape(n_days, K) ** 2, axis = 1).reshape(-1, 1)

        res = pd.DataFrame(vol, index=idx, columns=['Log_r'])

        return res, idx
    
    def match_idx(self, df, idx):

        df_copy = df.copy()
        df_copy.index = df_copy.index.to_period('D')
        idx = pd.to_datetime(idx).to_period('D')
        df_copy = df_copy.reindex(idx).dropna()
        return df_copy
    
    def get_trend(self, df):

        n_days = pd.Series(df.index.date).nunique()
        K = len(df) // n_days  

        Y = np.asarray(df)  

        trend = Y.reshape(n_days, K).mean(axis=0)

        return trend

    def get_detrended_series(self, df):
        '''
        Returns the detrended series of log (raw/squared/abs) returns.
        '''

        idx = pd.Series(df.index)

        n_days = pd.Series(df.index.date).nunique()
        K = len(df) // n_days 

        Y = np.asarray(df)

        trend = self.get_trend(df)

        X = Y.reshape(n_days, K) - trend

        res = pd.DataFrame(X.ravel(), index=idx, columns=['Log_r'])

        return res

    def clean_data(self):
        '''
        Limits series length for secondly if specifiied. 
        Sorts data into trading hours.
        Applies square, abs, vol calculation if requested. 
        Detrends any secondly data.
        '''

        # limit secondly data to fewer samples - change as needed
        length_s5 = len(self.datas[2])

        # sort data into trading hours for secondly data, daily is already fine
        self.datas_clean = [data['Log_r'] for data in self.datas]
        self.datas_clean[2] = self.datas_clean[2].iloc[int(length_s5/self.fraction_of_s5):]
        self.datas_clean[3] = self.datas_clean[3].iloc[int(length_s5/self.fraction_of_s5):] # do not modify datas in place

        self.datas_clean[2] = self.filter_trading_hours(self.datas_clean[2], self.start_trading_hr_euro, self.end_trading_hr_euro)
        self.datas_clean[2] = self.remove_first_day_and_last(self.datas_clean[2])
        self.datas_clean[2] = self.check_complete_5s_intervals(self.datas_clean[2], self.start_trading_hr_euro, self.end_trading_hr_euro)
        self.datas_clean[3] = self.filter_trading_hours(self.datas_clean[3], self.start_trading_hr_sgd, self.end_trading_hr_sgd)
        self.datas_clean[3] = self.remove_first_day_and_last(self.datas_clean[3])
        self.datas_clean[3] = self.check_complete_5s_intervals(self.datas_clean[3], self.start_trading_hr_sgd, self.end_trading_hr_sgd)

        if self.unit_test:
            self.unit_test_trading_hours()

        # build series we are interested in
        if self.type == 'raw_log_returns':
            self.datas_clean[2] = self.get_detrended_series(self.datas_clean[2])
            self.datas_clean[3] = self.get_detrended_series(self.datas_clean[3])
        elif self.type == 'squared_returns':
            self.datas_clean = [data ** 2 for data in self.datas_clean]
            self.datas_clean[2] = self.get_detrended_series(self.datas_clean[2])
            self.datas_clean[3] = self.get_detrended_series(self.datas_clean[3])
        elif self.type == 'absolute_returns':
            self.datas_clean = [np.abs(data) for data in self.datas_clean]
            self.datas_clean[2] = self.get_detrended_series(self.datas_clean[2])
            self.datas_clean[3] = self.get_detrended_series(self.datas_clean[3])
        elif self.type == 'daily_intraday_volatility':
            self.datas_clean[2], idx = self.get_sum_squared_intraday_returns(self.datas_clean[2])
            self.datas_clean[3], _ = self.get_sum_squared_intraday_returns(self.datas_clean[3])
            self.datas_clean[0] = self.match_idx(self.datas_clean[0] ** 2, idx)
            self.datas_clean[1] = self.match_idx(self.datas_clean[1] ** 2, idx)

        return self.datas_clean
            
    
    def get_f_and_h(self, X, segments):
        '''
        Does MF-DFA with m for X a single time series path.
        '''

        F_qs = list()

        for q in self.qs:

            _, F = MFDFA(X, lag = segments, order = self.m, q=q)
            F_qs.append(F)

        generalised_H = np.zeros(len(self.qs))

        for i in range(len(F_qs)): 

            generalised_H[i] = get_generalised_hurst(F_q = F_qs[i], ss = segments)

        tao_q, alpha_q, f_alpha = get_mf_spectrum(generalised_H, self.qs)

        return generalised_H, alpha_q, f_alpha, tao_q
    

    def do_analysis(self):

        self.clean_data()
        df = self.datas_clean
        self.qs = np.arange(self.q_min, self.q_max + 1, 1)

        for i in range(2):
            s_max = self.datas_clean[i].shape[0] // 4
            self.segments_d = np.logspace(np.log10(self.s_min_d), np.log10(s_max), num=self.steps)
            self.segments_d = np.unique(np.round(self.segments_d)).astype(int)

            model_data = np.asarray(df[i])

            generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(model_data, self.segments_d)

            self.hursts.append(generalised_H)
            self.alphas.append(alpha_q)
            self.f_alphas.append(f_alpha)
            self.taos.append(tao_q)


        for i in range(2, 4):
            s_max = self.datas_clean[i].shape[0] // 4 

            if self.type == 'daily_intraday_volatility':
                self.s_min_s5 = self.s_min_d # incase s_min_s5 isnt changed when using daily intraday vol setting since this is a smaller series

            self.segments_s5 = np.logspace(np.log10(self.s_min_s5), np.log10(s_max), num=self.steps)
            self.segments_s5 = np.unique(np.round(self.segments_s5)).astype(int)

            model_data = np.asarray(df[i])

            generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(model_data, self.segments_s5)

            self.hursts.append(generalised_H)
            self.alphas.append(alpha_q)
            self.f_alphas.append(f_alpha)
            self.taos.append(tao_q)

        return self.hursts, self.alphas, self.f_alphas, self.taos, self.segments_d, self.segments_s5, self.qs
    
    def do_analysis_shuffled(self):

        self.clean_data()
        df = [data.sample(frac=1, replace=False, random_state = 22 ).reset_index(drop=True) for data in self.datas_clean]
        self.qs = np.arange(self.q_min, self.q_max + 1, 1)

        for i in range(2):
            s_max = self.datas_clean[i].shape[0] // 4
            self.segments_d = np.logspace(np.log10(self.s_min_d), np.log10(s_max), num=self.steps)
            self.segments_d = np.unique(np.round(self.segments_d)).astype(int)

            model_data = np.asarray(df[i])

            generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(model_data, self.segments_d)

            self.hursts_shuffled.append(generalised_H)
            self.alphas_shuffled.append(alpha_q)
            self.f_alphas_shuffled.append(f_alpha)
            self.taos_shuffled.append(tao_q)


        for i in range(2, 4):
            s_max = self.datas_clean[i].shape[0] // 4 

            if self.type == 'daily_intraday_volatility':
                self.s_min_s5 = self.s_min_d # incase s_min_s5 isnt changed when using daily intraday vol setting since this is a smaller series

            self.segments_s5 = np.logspace(np.log10(self.s_min_s5), np.log10(s_max), num=self.steps)
            self.segments_s5 = np.unique(np.round(self.segments_s5)).astype(int)

            model_data = np.asarray(df[i])

            generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(model_data, self.segments_s5)

            self.hursts_shuffled.append(generalised_H)
            self.alphas_shuffled.append(alpha_q)
            self.f_alphas_shuffled.append(f_alpha)
            self.taos_shuffled.append(tao_q)

        return self.hursts_shuffled, self.alphas_shuffled, self.f_alphas_shuffled, self.taos_shuffled, self.segments_d, self.segments_s5, self.qs
    
    def get_plots(self):
        '''
        Plot MFDFA results for MF spectrum. 
        '''

        colors = ['black', 'blue','green', 'red', 'orange', 'purple']
        markers = ['o', 's', '^', 'd', 'v', '<']

        if self.plots:

            if self.shuffle:
                self.do_analysis_shuffled()
                H = self.hursts_shuffled
                A = self.alphas_shuffled
                F = self.f_alphas_shuffled
                T = self.taos_shuffled
            else:
                self.do_analysis()
                H = self.hursts
                A = self.alphas
                F = self.f_alphas
                T = self.taos

            plt.figure(figsize=(18, 12))

            # MF spectrum with error bars
            plt.subplot(2, 3, 1)
            q0_idx = np.where(self.qs == 0)[0][0]
            for i in range(len(self.models)):
                plt.plot(A[i], F[i], "-", color=colors[i], marker=markers[i], label=self.models[i], alpha = 0.5)
                plt.axvline(x=A[i][q0_idx], color=colors[i], linestyle='--', alpha=0.5)
            plt.xlabel(r'Singularity Strength, $\alpha$', fontsize=14)
            plt.ylabel(r'Multifractal Spectrum, $f(\alpha)$', fontsize=14)
            plt.title('Multifractal Spectrum', fontsize=16)
            plt.legend(fontsize='medium')
            plt.grid(True, alpha=0.3)

            # Hurst exponent
            plt.subplot(2, 3, 2)
            for i in range(len(self.models)):
                plt.plot(self.qs, H[i], "-", color=colors[i], marker=markers[i], label=self.models[i], alpha = 0.5)
            plt.xlabel(r'Order, $q$', fontsize=14)
            plt.ylabel(r'Hurst Exponent, $H(q)$', fontsize=14)
            plt.title('Hurst Exponent', fontsize=16)
            plt.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='H=0.5')
            plt.legend(fontsize='medium')
            plt.grid(True, alpha=0.3)

            #tao 
            plt.subplot(2, 3, 3)
            for i in range(len(self.models)):
                plt.plot(self.qs, T[i], "-", color=colors[i], marker=markers[i], label=self.models[i], alpha = 0.5)
            plt.xlabel(r'Order, $q$', fontsize=14)
            plt.ylabel(r'$\tau(q)$', fontsize=14)
            plt.title(r'$\tau(q)$', fontsize=16)
            plt.legend(fontsize='medium')
            plt.grid(True, alpha=0.3)

            # width of the spectra
            plt.subplot(2, 3, 4)
            for i in range(len(self.models)):
                A_finite = A[i][np.isfinite(A[i])]
                if len(A_finite) > 0:
                    widths = np.max(A_finite) - np.min(A_finite)
                    plt.scatter(i, widths, label=self.models[i], color=colors[i], marker=markers[i], s=100, alpha=0.5)
            plt.ylabel(r'$\alpha_{\max} - \alpha_{\min}$', fontsize=14)
            plt.title('Width of Multifractal Spectrum', fontsize=16)
            plt.tick_params(axis='x', labelsize=14)
            plt.legend(fontsize='large')
            plt.grid(True, alpha=0.3)

            # skewness of the spectra
            plt.subplot(2, 3, 5)
            q0_idx = np.where(self.qs == 0)[0][0]

            for j in range(len(self.models)):
                alpha_0 = A[j][q0_idx]
                A_finite = A[j][np.isfinite(A[j])]
                if len(A_finite) > 0 and np.isfinite(alpha_0):
                    alpha_max = np.max(A_finite)
                    alpha_min = np.min(A_finite)
                    if alpha_0 != alpha_min:
                        skewness = (alpha_max - alpha_0) / (alpha_0 - alpha_min)
                        plt.scatter(j, skewness, label=self.models[j], color=colors[j], marker=markers[j], s=100, alpha=0.5)

            plt.ylabel('Skewness', fontsize=14)
            plt.title('Skewness of Multifractal Spectrum', fontsize=16)
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Un-skewed')
            plt.tick_params(axis='x', labelsize=14)
            plt.legend(fontsize='large')
            plt.grid(True, alpha=0.3)

            # H(2)
            plt.subplot(2, 3, 6)
            q2_idx = np.where(self.qs == 2)[0][0]

            for i in range(len(self.models)):
                h2 = H[i][q2_idx]
                plt.scatter(i, h2, label=self.models[i], color=colors[i], marker=markers[i], s=100, alpha=0.5)
            plt.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='H=0.5')
            plt.ylabel(r'H(2)', fontsize=14)
            plt.title('Distribution of Hurst Exponents', fontsize=16)
            plt.tick_params(axis='x', labelsize=14)
            plt.legend(fontsize='large')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.suptitle(f'MFDFA Empirical Analysis - {self.type} - m={self.m} - {"Shuffled" if self.shuffle else "Original"}', fontsize=18, fontweight='bold')
            plt.subplots_adjust(top=0.91) 
            if self.shuffle:
                plt.savefig(f'/Users/alexvillamartin/Documents/MSc Diss/plots_new/MFDFA_Analysis_Empirical_{self.type}_{self.m}_shuffle.png')
            else:
                plt.savefig(f'/Users/alexvillamartin/Documents/MSc Diss/plots_new/MFDFA_Analysis_Empirical_{self.type}_{self.m}.png')
            plt.show()

    def compare_to_shuffle(self):
        '''
        Compare MFDFA results to shuffled data.
        '''

        if self.shuffle and self.compare_to_shuffle_ind:
            self.do_analysis()
            A = self.alphas

            self.do_analysis_shuffled()
            A_shuffled = self.alphas_shuffled

        widths_original = []
        widths_shuffled = []
        
        for j in range(len(self.models)):
                A_finite_og = A[j][np.isfinite(A[j])]
                A_finite_sh = A_shuffled[j][np.isfinite(A_shuffled[j])]
                
                if len(A_finite_og) > 0:
                    alpha_max_og = np.max(A_finite_og)
                    alpha_min_og = np.min(A_finite_og)
                    widths_original.append(alpha_max_og - alpha_min_og)
                else:
                    widths_original.append(0)

                if len(A_finite_sh) > 0:
                    alpha_max_sh = np.max(A_finite_sh)
                    alpha_min_sh = np.min(A_finite_sh)
                    widths_shuffled.append(alpha_max_sh - alpha_min_sh)
                else:
                    widths_shuffled.append(0)

        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(self.models))
        width = 0.35  

        bars1 = plt.bar(x_pos - width/2, widths_original, width, label='Original Data', alpha=0.8, color='darkgreen')
        bars2 = plt.bar(x_pos + width/2, widths_shuffled, width, label='Shuffled Data', alpha=0.8, color='crimson')

        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel(r'Spectrum Width', fontsize=12)
        plt.title(f'Multifractal Spectrum Width Comparison - {self.type}', fontsize=14)
        plt.xticks(x_pos, [name for name in self.models], rotation=0, ha='center')
        plt.legend(fontsize = 'large')
        plt.grid(True, alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'/Users/alexvillamartin/Documents/MSc Diss/plots_new/MFDFA_Analysis_Empirical_{self.type}_{self.m}_Shuffle_Comparison.png')
        plt.show()



    def do_analysis_surrogates(self):

        self.clean_data()
        df = [data for data in self.datas_clean]
        self.qs = np.arange(self.q_min, self.q_max + 1, 1)

        for i in range(len(self.models)):
            curr_data = np.asarray(df[i]).flatten()
            curr_surrogates = np.zeros((curr_data.shape[0], self.n_surrogates))

            print(f"Generating surrogates for model {self.models[i]}")

            curr_surrogates = surrogates(curr_data, ns=self.n_surrogates, tol_pc=self.accuracy_surrogates, verbose=True).T

            self.datas_clean_surrogates.append(curr_surrogates)


        for i in range(2):
            s_max = self.datas_clean[i].shape[0] // 4
            self.segments_d = np.logspace(np.log10(self.s_min_d), np.log10(s_max), num=self.steps)
            self.segments_d = np.unique(np.round(self.segments_d)).astype(int)

            model_data = np.asarray(self.datas_clean_surrogates[i])
            model_hursts = []
            model_alphas = []
            model_f_alphas = []
            model_taos = []

            for j in range(self.n_surrogates):
                surrogate_data = model_data[:, j]
                generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(surrogate_data, self.segments_d)

                model_hursts.append(generalised_H)
                model_alphas.append(alpha_q)
                model_f_alphas.append(f_alpha)
                model_taos.append(tao_q)

            self.hursts_surrogates.append(model_hursts)
            self.alphas_surrogates.append(model_alphas)
            self.f_alphas_surrogates.append(model_f_alphas)
            self.taos_surrogates.append(model_taos)


        for i in range(2, 4):
            s_max = self.datas_clean[i].shape[0] // 4 

            if self.type == 'daily_intraday_volatility':
                self.s_min_s5 = self.s_min_d # incase s_min_s5 isnt changed when using daily intraday vol setting since this is a smaller series

            self.segments_s5 = np.logspace(np.log10(self.s_min_s5), np.log10(s_max), num=self.steps)
            self.segments_s5 = np.unique(np.round(self.segments_s5)).astype(int)

            model_data = np.asarray(self.datas_clean_surrogates[i])
            model_hursts = []
            model_alphas = []
            model_f_alphas = []
            model_taos = []

            for j in range(self.n_surrogates):
                surrogate_data = model_data[:, j]
                generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(surrogate_data, self.segments_s5)

                model_hursts.append(generalised_H)
                model_alphas.append(alpha_q)
                model_f_alphas.append(f_alpha)
                model_taos.append(tao_q)

            self.hursts_surrogates.append(model_hursts)
            self.alphas_surrogates.append(model_alphas)
            self.f_alphas_surrogates.append(model_f_alphas)
            self.taos_surrogates.append(model_taos)

        return self.hursts_surrogates, self.alphas_surrogates, self.f_alphas_surrogates, self.taos_surrogates, self.segments_d, self.segments_s5, self.qs

    def surrogate_tests(self):
        '''
        Perform surrogate tests to assess the significance of the MF present. This breaks nonlinear correlations and 
        preserves linear and fat tails, both of which can contribute to MF so this ensures we test to see if there is a true
        difference attributed to multifractlity.

        Null is that the MF spectrum width of the original data is not significantly different from the surrogate data, hence no significant
        MF present.
        '''

        self.do_analysis_surrogates()
        self.do_analysis()

        for j in range(len(self.models)):
            alphas_model = self.alphas[j]
            alphas_surrogates_model = self.alphas_surrogates[j]

            alphas_finite = alphas_model[np.isfinite(alphas_model)]
            if len(alphas_finite) > 0:
                width_original = np.max(alphas_finite) - np.min(alphas_finite)
            else:
                width_original = 0
                
            alphas_surr_finite = alphas_surrogates_model.copy()
            alphas_surr_finite[~np.isfinite(alphas_surr_finite)] = np.nan
            widths_surrogates = np.nanmax(alphas_surr_finite, axis=1) - np.nanmin(alphas_surr_finite, axis=1)
            widths_surrogates = widths_surrogates[np.isfinite(widths_surrogates)]

            if len(widths_surrogates) > 0:
                p_val = ((widths_surrogates >= width_original).sum() + 1) / (len(widths_surrogates) + 1)
            else:
                p_val = np.nan
            self.p_values.append(np.array(p_val))
        
        return self.p_values, self.hursts_surrogates, self.alphas_surrogates, self.f_alphas_surrogates, self.taos_surrogates, self.datas_clean_surrogates
    


class MFDFA_Analysis_Empirical_D_only:
    '''
    Class to perform MFDFA analysis on a type of returns / vol dataframe for empirical data with daily / S5 data
    and selecting relevant trading hours and detrending. Unlike in simulation one we do not compute block vol yet and instead
    compute daily vol from 5 second data and compare this to daily squared returns. Add block vol later if needed.

    Note: Daily intraday vol of secondly data is compared to squared daily returns of daily data in the spectrum plots.  
    '''

    def __init__(self, 
                datas: list, 
                fraction_of_s5: float, 
                s_min_d: int, 
                s_min_s5: int,
                q_min: int, 
                q_max: int,
                ms: np.array, 
                steps: int, 
                type: str, 
                models: list, 
                plots: bool, 
                end_trading_hr_euro: int,
                end_trading_hr_sgd: int, 
                start_trading_hr_euro: int,
                start_trading_hr_sgd: int, 
                unit_test: bool, 
                shuffle: bool, 
                compare_to_shuffle_ind: bool, 
                n_surrogates: int, 
                accuracy_surrogates: float) -> None: # doesnt return anything
        '''
        Parameters:
        - datas: List containing the raw returns/vol to analyse. Index order is same as models order. 
        - fraction_of_s5: Fraction of 5 second data to use for analysis (e.g. 10 means every 10th sample).
        - s_min_d: Minimum scale for daily data.
        - s_min_s5: Minimum scale for 5 second data.
        - q_min: Minimum order for generalized Hurst exponent.
        - q_max: Maximum order for generalized Hurst exponent.
        - m: Polynomial fitting for MFDFA.
        - steps: Number of steps for the segment array.
        - type: Type of data passed to the MFDFA analysis (squared, raw, volatility).
        - models: List of model names for the data.
        - plots: Boolean to indicate if plots should be generated.
        - end_trading_hr_euro: End trading hour for Euro data.
        - end_trading_hr_sgd: End trading hour for SGD data.
        - start_trading_hr_euro: Start trading hour for Euro data.
        - start_trading_hr_sgd: Start trading hour for SGD data.
        - unit_test: Boolean to indicate if unit tests should be run.
        - shuffle: Boolean to indicate if data should be shuffled for analysis.
        - compare_to_shuffled: Boolean to indicate if data should be compared to shuffled version for analysis.
        - n_surrogates: Number of surrogates to generate for surrogate tests.
        - accuracy_surrogates: Accuracy for surrogate tests.
        '''

        self.datas = datas
        self.fraction_of_s5 = fraction_of_s5
        self.s_min_d = s_min_d
        self.s_min_s5 = s_min_s5
        self.q_min = q_min
        self.q_max = q_max
        self.ms = ms
        self.steps = steps
        self.type = type
        self.models = models
        self.plots = plots
        self.end_trading_hr_euro = end_trading_hr_euro
        self.end_trading_hr_sgd = end_trading_hr_sgd
        self.start_trading_hr_euro = start_trading_hr_euro
        self.start_trading_hr_sgd = start_trading_hr_sgd
        self.unit_test = unit_test
        self.shuffle = shuffle
        self.compare_to_shuffle_ind = compare_to_shuffle_ind
        self.n_surrogates = n_surrogates
        self.accuracy_surrogates = accuracy_surrogates

        self.datas_clean = None
        self.segments_d = None
        self.segments_s5 = None
        self.qs = None
        self.s_max = None
        self.hursts = []
        self.alphas = []
        self.f_alphas = []
        self.taos = []
        self.datas_clean_shuffled = None
        self.hursts_shuffled = []
        self.alphas_shuffled = []
        self.f_alphas_shuffled = []
        self.taos_shuffled = []
        self.datas_clean_surrogates = []
        self.hursts_surrogates = []
        self.alphas_surrogates = []
        self.f_alphas_surrogates = []
        self.taos_surrogates = []
        self.p_values = []

    def filter_trading_hours(self, df, start_hour, end_hour):
        '''
        Make sure we select the trading hours we are interested in.
        '''

        df_copy = df.copy()
        
        mask = (df_copy.index.time >= pd.Timestamp(f'{start_hour:02d}:00:05').time()) & (df_copy.index.time <= pd.Timestamp(f'{end_hour:02d}:00:00').time())
        
        return df_copy[mask]

    def remove_first_day_and_last(self, df):
        '''
        Remove first and last day incase they dont start at the same time as the rest of the data.
        '''

        df_copy = df.copy()

        first_date = df_copy.index.date[0] 
        last_date = df_copy.index.date[-1]

        mask = (df_copy.index.date != first_date) & (df_copy.index.date != last_date)

        return df_copy[mask]
    
    def check_complete_5s_intervals(self, df, start_hour, end_hour):
        '''
        This function overlays a full 5 second index onto data and ffils missing values to ensure all intervals in 
        specified hours are there. 
        '''

        df_copy = df.copy()
        
        start_date = df_copy.index.min().floor('D')
        end_date = df_copy.index.max().ceil('D')
        full_index = pd.date_range(start=start_date, end=end_date, freq='5s')
        
        trading_mask = (full_index.time >= pd.Timestamp(f'{start_hour:02d}:00:05').time()) & (full_index.time <= pd.Timestamp(f'{end_hour:02d}:00:00').time())
        weekday_mask = full_index.dayofweek < 5 

        trading_index = full_index[weekday_mask & trading_mask]
        
        df_complete = df_copy.reindex(trading_index).ffill().bfill() # in case of gaps at start
        
        return df_complete
    
    def unit_test_trading_hours(self):
        '''
        Unit test to check if trading hours are correctly filtered.
        '''
        euro_s5_clean = self.datas_clean[2]
        sgd_s5_clean = self.datas_clean[3]

        print("Euro S5 data info:")
        print(f"Start: {euro_s5_clean.index.min()}")
        print(f"End: {euro_s5_clean.index.max()}")
        print(f"Total observations: {len(euro_s5_clean)}")
        print(f"Expected 5 second intervals per day (9 hours): {9 * 60 * 60 / 5}")
        print(f"Number of trading days: {len(pd.Series(euro_s5_clean.index.date).unique())}")
        print(f"Expected total observations: {len(pd.Series(euro_s5_clean.index.date).unique()) * 9 * 60 * 60 / 5}")

        print("\nSGD S5 data info:")
        print(f"Start: {sgd_s5_clean.index.min()}")
        print(f"End: {sgd_s5_clean.index.max()}")
        print(f"Total observations: {len(sgd_s5_clean)}")
        print(f"Number of trading days: {len(pd.Series(sgd_s5_clean.index.date).unique())}")

        print(f"\nEuro missing values: {euro_s5_clean.isnull().sum().sum()}")
        print(f"SGD missing values: {sgd_s5_clean.isnull().sum().sum()}")

    def get_sum_squared_intraday_returns(self, df):
        '''
        RV = \sum_{i=1}^{n} r_i^2 where n is number of returns in the day.
        '''
        idx = pd.Series(df.index.date).unique()
        n_days = pd.Series(df.index.date).nunique()
        K = len(df) // n_days  # number of returns per day
        
        vol = np.sum(df.values.reshape(n_days, K) ** 2, axis = 1).reshape(-1, 1)

        res = pd.DataFrame(vol, index=idx, columns=['Log_r'])

        return res, idx
    
    def match_idx(self, df, idx):

        df_copy = df.copy()
        df_copy.index = df_copy.index.to_period('D')
        idx = pd.to_datetime(idx).to_period('D')
        df_copy = df_copy.reindex(idx).dropna()
        return df_copy
    
    def get_trend(self, df):

        n_days = pd.Series(df.index.date).nunique()
        K = len(df) // n_days  

        Y = np.asarray(df)  

        trend = Y.reshape(n_days, K).mean(axis=0)

        return trend

    def get_detrended_series(self, df):
        '''
        Returns the detrended series of log (raw/squared/abs) returns.
        '''

        idx = pd.Series(df.index)

        n_days = pd.Series(df.index.date).nunique()
        K = len(df) // n_days 

        Y = np.asarray(df)

        trend = self.get_trend(df)

        X = Y.reshape(n_days, K) - trend

        res = pd.DataFrame(X.ravel(), index=idx, columns=['Log_r'])

        return res

    def clean_data(self):
        '''
        Limits series length for secondly if specifiied. 
        Sorts data into trading hours.
        Applies square, abs, vol calculation if requested. 
        Detrends any secondly data.
        '''

        # sort data into trading hours for secondly data, daily is already fine
        self.datas_clean = [data['Log_r'] for data in self.datas]
        
        if self.unit_test:
            self.unit_test_trading_hours()

 
        if self.type == 'squared returns':
            self.datas_clean = [data ** 2 for data in self.datas_clean]

        elif self.type == 'absolute returns':
            self.datas_clean = [np.abs(data) for data in self.datas_clean]

        return self.datas_clean
            
    
    def get_f_and_h(self, X, segments, m):
        '''
        Does MF-DFA with m for X a single time series path.
        '''

        F_qs = list()

        for q in self.qs:

            _, F = MFDFA(X, lag = segments, order = m, q=q)
            F_qs.append(F)

        generalised_H = np.zeros(len(self.qs))

        for i in range(len(F_qs)): 

            generalised_H[i] = get_generalised_hurst(F_q = F_qs[i], ss = segments)

        tao_q, alpha_q, f_alpha = get_mf_spectrum(generalised_H, self.qs)

        return generalised_H, alpha_q, f_alpha, tao_q
    

    def do_analysis(self):

        self.clean_data()
        df = self.datas_clean
        self.qs = np.arange(self.q_min, self.q_max + 1, 1)

        for i in range(len(self.models)):
            s_max = self.datas_clean[i].shape[0] // 4
            self.segments_d = np.logspace(np.log10(self.s_min_d), np.log10(s_max), num=self.steps)
            self.segments_d = np.unique(np.round(self.segments_d)).astype(int)

            model_data = np.asarray(df[i])

            m_model = self.ms[i]

            generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(model_data, self.segments_d, m_model)

            self.hursts.append(generalised_H)
            self.alphas.append(alpha_q)
            self.f_alphas.append(f_alpha)
            self.taos.append(tao_q)

        return self.hursts, self.alphas, self.f_alphas, self.taos, self.segments_d, self.segments_s5, self.qs
    
    def do_analysis_shuffled(self):

        self.clean_data()
        df = [data.sample(frac=1, replace=False, random_state = 22 ).reset_index(drop=True) for data in self.datas_clean]
        self.qs = np.arange(self.q_min, self.q_max + 1, 1)

        for i in range(2):
            s_max = self.datas_clean[i].shape[0] // 10
            self.segments_d = np.logspace(np.log10(self.s_min_d), np.log10(s_max), num=self.steps)
            self.segments_d = np.unique(np.round(self.segments_d)).astype(int)

            model_data = np.asarray(df[i])

            generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(model_data, self.segments_d)

            self.hursts_shuffled.append(generalised_H)
            self.alphas_shuffled.append(alpha_q)
            self.f_alphas_shuffled.append(f_alpha)
            self.taos_shuffled.append(tao_q)


        for i in range(2, 4):
            s_max = self.datas_clean[i].shape[0] // 4 

            if self.type == 'daily_intraday_volatility':
                self.s_min_s5 = self.s_min_d # incase s_min_s5 isnt changed when using daily intraday vol setting since this is a smaller series

            self.segments_s5 = np.logspace(np.log10(self.s_min_s5), np.log10(s_max), num=self.steps)
            self.segments_s5 = np.unique(np.round(self.segments_s5)).astype(int)

            model_data = np.asarray(df[i])

            generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(model_data, self.segments_s5)

            self.hursts_shuffled.append(generalised_H)
            self.alphas_shuffled.append(alpha_q)
            self.f_alphas_shuffled.append(f_alpha)
            self.taos_shuffled.append(tao_q)

        return self.hursts_shuffled, self.alphas_shuffled, self.f_alphas_shuffled, self.taos_shuffled, self.segments_d, self.segments_s5, self.qs
    
    def get_plots(self):
        '''
        Plot MFDFA results for MF spectrum. 
        '''

        colors = ['black', 'blue','green', 'red', 'orange', 'purple']
        markers = ['o', 's', '^', 'd', 'v', '<']

        if self.plots:

            if self.shuffle:
                self.do_analysis_shuffled()
                H = self.hursts_shuffled
                A = self.alphas_shuffled
                F = self.f_alphas_shuffled
                T = self.taos_shuffled
            else:
                self.do_analysis()
                H = self.hursts
                A = self.alphas
                F = self.f_alphas
                T = self.taos

            plt.style.use('seaborn-v0_8-dark')

            plt.figure(figsize=(18, 12))

            # MF spectrum with error bars
            plt.subplot(2, 3, 1)
            q0_idx = np.where(self.qs == 0)[0][0]

            for i in range(len(self.models)):
                mask_pos_i = F[i] > 0
                plt.plot(A[i][mask_pos_i], F[i][mask_pos_i], "-", color=colors[i], marker=markers[i], label=self.models[i], alpha = 0.5)
                plt.axvline(x=A[i][q0_idx], color=colors[i], linestyle='--', alpha=0.5)
            plt.xlabel(r'Singularity Strength, $\alpha$', fontsize=15)
            plt.ylabel(r'Multifractal Spectrum, $f(\alpha)$', fontsize=15)
            plt.title('Multifractal Spectrum', fontsize=18)
            plt.legend(fontsize='medium')
            plt.grid(True, alpha=0.7)

            # Hurst exponent
            plt.subplot(2, 3, 2)
            for i in range(len(self.models)):
                plt.plot(self.qs, H[i], "-", color=colors[i], marker=markers[i], label=self.models[i], alpha = 0.5)
            plt.xlabel(r'Order, $q$', fontsize=15)
            plt.ylabel(r'Hurst Exponent, $h(q)$', fontsize=15)
            plt.title('Generalised Hurst Exponent', fontsize=18)
            plt.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='H=0.5')
            plt.legend(fontsize='medium')
            plt.grid(True, alpha=0.7)

            #tao 
            plt.subplot(2, 3, 3)
            for i in range(len(self.models)):
                plt.plot(self.qs, T[i], "-", color=colors[i], marker=markers[i], label=self.models[i], alpha = 0.5)
            plt.xlabel(r'Order, $q$', fontsize=15)
            plt.ylabel(r'$\tau(q)$', fontsize=15)
            plt.title(r'$\tau(q)$', fontsize=18)
            plt.legend(fontsize='medium')
            plt.grid(True, alpha=0.7)

            # width of the spectra
            plt.subplot(2, 3, 4)
            for i in range(len(self.models)):
                A_finite = A[i][np.isfinite(A[i])]
                if len(A_finite) > 0:
                    widths = np.max(A_finite) - np.min(A_finite)
                    plt.scatter(i, widths, label=self.models[i], color=colors[i], marker=markers[i], s=100, alpha=0.5)
            plt.ylabel(r'$\alpha_{\max} - \alpha_{\min}$', fontsize=15)
            plt.title('Width of Multifractal Spectrum', fontsize=18)
            plt.tick_params(axis='x', labelsize=14)
            plt.legend(fontsize='large')
            plt.grid(True, alpha=0.7)

            # skewness of the spectra
            plt.subplot(2, 3, 5)
            q0_idx = np.where(self.qs == 0)[0][0]

            for j in range(len(self.models)):
                alpha_0 = A[j][q0_idx]
                A_finite = A[j][np.isfinite(A[j])]
                if len(A_finite) > 0 and np.isfinite(alpha_0):
                    alpha_max = np.max(A_finite)
                    alpha_min = np.min(A_finite)
                    if alpha_0 != alpha_min:
                        skewness = (alpha_max - alpha_0) / (alpha_0 - alpha_min)
                        plt.scatter(j, skewness, label=self.models[j], color=colors[j], marker=markers[j], s=100, alpha=0.5)

            plt.ylabel('Skewness', fontsize=15)
            plt.title('Skewness of Multifractal Spectrum', fontsize=18)
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Un-skewed')
            plt.tick_params(axis='x', labelsize=14)
            plt.legend(fontsize='large')
            plt.grid(True, alpha=0.7)

            # H(2)
            plt.subplot(2, 3, 6)
            q2_idx = np.where(self.qs == 2)[0][0]

            for i in range(len(self.models)):
                h2 = H[i][q2_idx]
                plt.scatter(i, h2, label=self.models[i], color=colors[i], marker=markers[i], s=100, alpha=0.5)
            plt.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='H=0.5')
            plt.ylabel(r'H(2)', fontsize=15)
            plt.title('Distribution of Hurst Exponents', fontsize=18)
            plt.tick_params(axis='x', labelsize=14)
            plt.legend(fontsize='large', loc='lower right')
            plt.grid(True, alpha=0.7)

            plt.tight_layout()
            plt.suptitle(f'MFDFA Empirical Analysis - {self.type}', fontsize=22, fontweight='bold')
            #plt.suptitle(f'MFDFA Empirical Analysis - {self.type} - m={self.m} - {"Shuffled" if self.shuffle else "Original"}', fontsize=18, fontweight='bold')
            plt.subplots_adjust(top=0.91) 
            if self.shuffle:
                plt.savefig(f'/Users/alexvillamartin/Documents/MSc Diss/plots_aug/MFDFA_Analysis_Empirical_{self.type}_{self.m}_shuffle.png')
            else:
                plt.savefig(f'/Users/alexvillamartin/Documents/MSc Diss/plots_aug/MFDFA_Analysis_Empirical_{self.type}.png')
            plt.show()


    def compare_to_shuffle(self):
        '''
        Compare MFDFA results to shuffled data.
        '''

        if self.shuffle and self.compare_to_shuffle_ind:
            self.do_analysis()
            A = self.alphas

            self.do_analysis_shuffled()
            A_shuffled = self.alphas_shuffled

        widths_original = []
        widths_shuffled = []
        
        for j in range(len(self.models)):
                A_finite_og = A[j][np.isfinite(A[j])]
                A_finite_sh = A_shuffled[j][np.isfinite(A_shuffled[j])]
                
                if len(A_finite_og) > 0:
                    alpha_max_og = np.max(A_finite_og)
                    alpha_min_og = np.min(A_finite_og)
                    widths_original.append(alpha_max_og - alpha_min_og)
                else:
                    widths_original.append(0)

                if len(A_finite_sh) > 0:
                    alpha_max_sh = np.max(A_finite_sh)
                    alpha_min_sh = np.min(A_finite_sh)
                    widths_shuffled.append(alpha_max_sh - alpha_min_sh)
                else:
                    widths_shuffled.append(0)

        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(self.models))
        width = 0.35  

        bars1 = plt.bar(x_pos - width/2, widths_original, width, label='Original Data', alpha=0.8, color='darkgreen')
        bars2 = plt.bar(x_pos + width/2, widths_shuffled, width, label='Shuffled Data', alpha=0.8, color='crimson')

        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel(r'Spectrum Width', fontsize=12)
        plt.title(f'Multifractal Spectrum Width Comparison - {self.type}', fontsize=14)
        plt.xticks(x_pos, [name for name in self.models], rotation=0, ha='center')
        plt.legend(fontsize = 'large')
        plt.grid(True, alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'/Users/alexvillamartin/Documents/MSc Diss/plots_new/MFDFA_Analysis_Empirical_{self.type}_{self.m}_Shuffle_Comparison.png')
        plt.show()



    def do_analysis_surrogates(self):

        self.clean_data()
        df = [data for data in self.datas_clean]
        self.qs = np.arange(self.q_min, self.q_max + 1, 1)

        for i in range(len(self.models)):
            curr_data = np.asarray(df[i]).flatten()
            curr_surrogates = np.zeros((curr_data.shape[0], self.n_surrogates))

            print(f"Generating surrogates for model {self.models[i]}")

            curr_surrogates = surrogates(curr_data, ns=self.n_surrogates, tol_pc=self.accuracy_surrogates, verbose=True).T

            self.datas_clean_surrogates.append(curr_surrogates)


        for i in range(len(self.models)):
            s_max = self.datas_clean[i].shape[0] // 4
            self.segments_d = np.logspace(np.log10(self.s_min_d), np.log10(s_max), num=self.steps)
            self.segments_d = np.unique(np.round(self.segments_d)).astype(int)

            model_data = np.asarray(self.datas_clean_surrogates[i])
            model_hursts = []
            model_alphas = []
            model_f_alphas = []
            model_taos = []

            for j in range(self.n_surrogates):
                surrogate_data = model_data[:, j]
                generalised_H, alpha_q, f_alpha, tao_q = self.get_f_and_h(surrogate_data, self.segments_d)

                model_hursts.append(generalised_H)
                model_alphas.append(alpha_q)
                model_f_alphas.append(f_alpha)
                model_taos.append(tao_q)

            self.hursts_surrogates.append(model_hursts)
            self.alphas_surrogates.append(model_alphas)
            self.f_alphas_surrogates.append(model_f_alphas)
            self.taos_surrogates.append(model_taos)

        return self.hursts_surrogates, self.alphas_surrogates, self.f_alphas_surrogates, self.taos_surrogates, self.segments_d, self.segments_s5, self.qs

    def surrogate_tests(self):
        '''
        Perform surrogate tests to assess the significance of the MF present. This breaks nonlinear correlations and 
        preserves linear and fat tails, both of which can contribute to MF so this ensures we test to see if there is a true
        difference attributed to multifractlity.

        Null is that the MF spectrum width of the original data is not significantly different from the surrogate data, hence no significant
        MF present.
        '''

        self.do_analysis_surrogates()
        self.do_analysis()

        for j in range(len(self.models)):
            alphas_model = self.alphas[j]
            alphas_surrogates_model = self.alphas_surrogates[j]

            alphas_finite = alphas_model[np.isfinite(alphas_model)]
            if len(alphas_finite) > 0:
                width_original = np.max(alphas_finite) - np.min(alphas_finite)
            else:
                width_original = 0
                
            alphas_surr_finite = alphas_surrogates_model.copy()
            alphas_surr_finite = np.asarray(alphas_surrogates_model, dtype=float)
            alphas_surr_finite[~np.isfinite(alphas_surr_finite)] = np.nan
            widths_surrogates = np.nanmax(alphas_surr_finite, axis=1) - np.nanmin(alphas_surr_finite, axis=1)
            widths_surrogates = widths_surrogates[np.isfinite(widths_surrogates)]

            if len(widths_surrogates) > 0:
                p_val = ((widths_surrogates >= width_original).sum() + 1) / (len(widths_surrogates) + 1)
            else:
                p_val = np.nan
            self.p_values.append(np.array(p_val))
        
        return self.p_values, self.hursts_surrogates, self.alphas_surrogates, self.f_alphas_surrogates, self.taos_surrogates, self.datas_clean_surrogates
    