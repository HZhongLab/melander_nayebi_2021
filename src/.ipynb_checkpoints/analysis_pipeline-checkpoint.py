import numpy as np
import matplotlib.pyplot as plt
import os, copy
import fnmatch
from scipy.stats import sem, pearsonr, linregress, norm, shapiro
# from main_utils import Dendrite
from markov_utils import (
    markov_pipeline,
    bootstrap_markov_pipeline,
    compute_bootstrap_mean_std,
    bin_clip_func,
    bin_equal_addmult,
    compute_conditional_mean_var,
    verify_detailed_balance,
    compute_conditional_std_unbinned
)
from markov_utils import (
    cross_correlation,
    compute_velocity,
    compute_deltaw,
    run_chain,
    preproc_intensity,
    weight_triggered_average,
)
from figure_utils import save_figure
from distance_utils import create_dynamic_events_matrix

from analysis_defaults import (
    DEFAULT_DENDRITE_KWARGS,
    DEFAULT_BIN_KWARGS,
    DEFAULT_MARKOV_KWARGS,
    DEFAULT_BOOTSTRAP_MARKOV_KWARGS,
    DEFAULT_KESTEN_KWARGS
)

from kesten_utils import (
    linear_regress,
    run_kesten
)

class Analysis:
    """Analysis.
    """

    def __init__(
        self,
        cell_type,
        input_weight_matrix=None,
        data_path_prefix=None,
        data_path=None,
        user="aran_laptop",
        condition="control",
        weight_start_time_idx=None,
        fit_markov_init=True,
        fit_kesten_init=True,
        dendrite_kwargs=DEFAULT_DENDRITE_KWARGS,
        bin_kwargs=DEFAULT_BIN_KWARGS,
        markov_kwargs=DEFAULT_MARKOV_KWARGS,
        kesten_kwargs=DEFAULT_KESTEN_KWARGS,
        bootstrap_markov_kwargs=DEFAULT_BOOTSTRAP_MARKOV_KWARGS,
    ):
        """__init__.

        Parameters
        ----------
        data_path_prefix :
            data_path_prefix
        data_path :
            data_path
        user :
            user
        cell_type :
            cell_type
        condition :
            condition
        weight_start_time_idx :
            weight_start_time_idx
        fit_markov_init :
            fit_markov_init
        dendrite_kwargs :
            dendrite_kwargs
        bin_kwargs :
            bin_kwargs
        markov_kwargs :
            markov_kwargs
        bootstrap_markov_kwargs :
            bootstrap_markov_kwargs
        """
        if cell_type == 'pv':
            impath = '/mnt/raw_data/synvivo_data/hq_dendrites_8_TP_only/pv/pv_control/'
        elif cell_type == 'pyr':
            impath = '/mnt/raw_data/synvivo_data/hq_dendrites_8_TP_only/pyramidal/pyramidal_control/'
        else:
            raise ValueError

        self.data_path_prefix = data_path_prefix
        self.user = user
        self.cell_type = cell_type
        self.condition = condition
        self.weight_start_time_idx = weight_start_time_idx
        self.fit_markov_init = fit_markov_init
        self.fit_kesten_init = fit_kesten_init
        self.dendrite_kwargs = dendrite_kwargs
        self.bin_kwargs = bin_kwargs
        self.markov_kwargs = markov_kwargs
        self.bootstrap_markov_kwargs = bootstrap_markov_kwargs
        self.hq_only = self.dendrite_kwargs.get("high_quality_only", False)
        self.impath = impath
        self.data_path = data_path

        # weights and filenames should be set only once in object creation
        # so we return them in their respective helper functions and set these values in the init method
        if input_weight_matrix is None:
            self.filenames = self._load_filenames()
            self.agg_density = None
            if not self.hq_only:
                self.agg_density = self._aggregate_density()
            self.weights_by_dendrite = {}
            self.weights = self._aggregate_weights()
        else:
            self.weights = input_weight_matrix

        if self.weight_start_time_idx is not None:
            self.weights = self.weights[:, self.weight_start_time_idx :]

        self.log_weights = np.log(self.weights[self.weights > 0].flatten())

        self.dynamics_matrix = create_dynamic_events_matrix(
            preproc_intensity(self.weights)
        )
        self.markov_out = None
        self.bootstrap_markov_out = None
        if self.fit_markov_init:
            # this function sets the markov_out and bootstrap_markov_out objects in init
            # this function can be rerun later to overwrite the markov objects
            self.fit_markov_procedure()
        if self.fit_kesten_init:
            self.fit_kesten_procedure()
        self.chain_state_matrix = None
        self.chain_weight_matrix = None

    def _load_filenames(self):
        """_load_filenames.
        """
        filenames = []
        for dirpath, dirs, files in os.walk(self.data_path):
            for nm_f in fnmatch.filter(files, "*.mat"):
                filenames.append(nm_f)
        return filenames

    def _aggregate_weights(self):
        print(self.impath)
        """_aggregate_weights.
        """
        agg_weights = []
        # agg_spine_volumes = []
        for nm_f in self.filenames:
            try:
                print(nm_f)

                d = Dendrite(os.path.join(self.data_path, nm_f), impath=self.impath,**self.dendrite_kwargs)
                self.weights_by_dendrite[nm_f] = d.weights_nonan
                if (
                    self.cell_type == "pv"
                ):  # all the pv dendrites have data up to 7 observation days, not all have it for all 8 days
                    agg_weights.append(d.weights_nonan[:, :7])
                else:
                    agg_weights.append(d.weights_nonan)
            except:
                print('Exception {}'.format(nm_f))

        final_weights = np.concatenate(agg_weights, axis=0)

        return final_weights

    def _aggregate_density(self):
        """_aggregate_density.
        """
        assert self.hq_only is False
        self._density_keys = ["total_density", "spine_density", "shaft_density"]
        agg_density = {}
        for k in self._density_keys:
            agg_density[k] = []
        # aggregate across dendrites
        for nm_f in self.filenames:
            d = Dendrite(os.path.join(self.data_path, nm_f), **self.dendrite_kwargs)
            for k in self._density_keys:
                agg_density[k].append(d.binary_dynamics[k])

        for k in self._density_keys:
            # concatenate across synapses and days
            agg_density[k] = np.concatenate(agg_density[k], axis=0)
        return agg_density

    def plot_density(
        self,
        ax=None,
        standalone_fig=True,
        save_nm=None,
        fig_kwargs={},
        density_keys=None,
    ):
        """plot_density.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        density_keys :
            density_keys
        """
        assert (self.hq_only is False) and (self.agg_density is not None)
        if density_keys is None:
            density_keys = self._density_keys

        if standalone_fig:
            plt.clf()
        for k in density_keys:
            plt.bar(
                k,
                np.mean(self.agg_density[k]),
                yerr=sem(self.agg_density[k]),
                capsize=5,
            )

        if ax is None:
            ax = plt.subplot(111)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def plot_stationary(
        self,
        ax=None,
        standalone_fig=True,
        save_nm=None,
        fig_kwargs={},
        fit_gaussian=True,
        gaussian_lw=4.0,
        log=False,
    ):
        """plot_stationary.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        fit_gaussian :
            fit_gaussian
        gaussian_lw :
            gaussian_lw
        log :
            log
        """
        if not log:
            weights = np.concatenate(self.weights, axis=0)
            weights = weights[weights > 0]

        else:
            weights = self.log_weights

        if standalone_fig:
            plt.clf()
        plt.hist(weights, normed=True, bins="auto")
        if fit_gaussian:
            fit_mean, fit_std = norm.fit(weights)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            y = norm.pdf(x, fit_mean, fit_std)
            plt.plot(x, y, linewidth=gaussian_lw)
            plt.title("Mean {}, Std {}".format(fit_mean, fit_std))

        if ax is None:
            ax = plt.subplot(111)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()



    def plot_stationary_qq(
        self, ax=None, standalone_fig=True, save_nm=None, fig_kwargs={}
    ):
        """plot_stationary_qq.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """

        import statsmodels.api as sm

        if standalone_fig:
            plt.clf()

        fig = sm.qqplot(self.log_weights, line="45", fit=True)
        plt.title(
            "{} Shapiro p-value {}".format(self.cell_type, shapiro(self.log_weights)[1])
        )

        if ax is None:
            ax = plt.subplot(111)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def fit_markov_procedure(
        self, bin_kwargs=None, markov_kwargs=None, bootstrap_markov_kwargs=None
    ):
        """fit_markov_procedure.

        Parameters
        ----------
        bin_kwargs :
            bin_kwargs
        markov_kwargs :
            markov_kwargs
        bootstrap_markov_kwargs :
            bootstrap_markov_kwargs
        """

        if bin_kwargs is not None:
            self.bin_kwargs = bin_kwargs
        if markov_kwargs is not None:
            self.markov_kwargs = markov_kwargs
        if bootstrap_markov_kwargs is not None:
            self.bootstrap_markov_kwargs = bootstrap_markov_kwargs

        markov_bin_kwargs = {
            "bin_params": self.bin_kwargs.get(
                "bin_params", DEFAULT_BIN_KWARGS["bin_params"]
            ),
            "bin_per_day": self.bin_kwargs.get(
                "bin_per_day", DEFAULT_BIN_KWARGS["bin_per_day"]
            ),
        }

        self.markov_kwargs.update(markov_bin_kwargs)
        markov_out = markov_pipeline(self.weights, **self.markov_kwargs)

        if self.bin_kwargs.get("equal_bins", DEFAULT_BIN_KWARGS["equal_bins"]):
            # create bin_params equal bins for nonzero weights across all days and synapses then make last bin to extend to max value
            orig_markov_bins = markov_out["bins_touse"]
            print("Original bins", orig_markov_bins)
            new_markov_bins = bin_clip_func(orig_markov_bins, bin_clip_idx=self.bin_kwargs.get("bin_clip_idx", DEFAULT_BIN_KWARGS["bin_clip_idx"]))
            print("New bins", new_markov_bins)
            if 'bins_touse' in self.markov_kwargs.keys():
                self.markov_kwargs.pop('bins_touse')
            markov_out = markov_pipeline(
                self.weights, bins_touse=new_markov_bins, **self.markov_kwargs
            )

        self.markov_out = markov_out
        self.markov_bins = self.markov_out["bins_touse"]

        if self.bootstrap_markov_kwargs.get(
            "bootstrap", DEFAULT_BOOTSTRAP_MARKOV_KWARGS["bootstrap"]
        ):
            bootstrap_markov_kwargs = copy.deepcopy(self.bootstrap_markov_kwargs)
            bootstrap_markov_kwargs.pop("bootstrap", None)
            bootstrap_markov_kwargs.update(
                self.markov_kwargs
            )  # so that the bootstrap chain is always the same as the markov chain
            if 'bins_touse' in bootstrap_markov_kwargs.keys():
                bootstrap_markov_kwargs.pop('bins_touse')
                print('Bootstrap Markov bins', self.markov_bins)
            self.bootstrap_markov_out = bootstrap_markov_pipeline(
                self.weights, bins_touse=self.markov_bins, **bootstrap_markov_kwargs
            )

    def simulate_markov_chain(self,
                              **simulation_kwargs):
        """simulate_markov_chain.
        """
        # sample weights from the chain
        # you need to fit the chain first
        assert self.markov_out is not None
        chain_kwargs = copy.deepcopy(self.markov_kwargs)
        chain_kwargs.update({"bins_touse": self.markov_bins})
        self.chain_state_matrix, self.chain_weight_matrix = run_chain(
            dend=None,
            weights_agg=self.weights,
            markov_kwargs=chain_kwargs,
            **simulation_kwargs
        )

    def fit_kesten_procedure(
        self,
        fit_to_difference=True,
        kesten_bin_avg=None,
        kesten_bin_diff_avg=None,
        kesten_bin_diff_var=None,
        bin_size=None,
        weights_mat=None,
        preproc_intensity_mat=True,
        train_frac=1.0,
        fit_intercept=True
    ):

        self.kesten_a = None
        self.kesten_b = None
        self.kesten_c = None
        self.kesten_d = None
        if (kesten_bin_avg is None) or (kesten_bin_diff_avg is None) or (kesten_bin_diff_var is None):

            if bin_size is None:
                if self.cell_type == 'pyr':
                    bin_size = 165
                elif self.cell_type == 'pv':
                    bin_size = 125
                else:
                    raise ValueError
            self.bin_size = bin_size

            if weights_mat is None:
                weights_mat = self.weights
            else:
                print('Using inputted weight mat')
            if preproc_intensity_mat:
                weights_mat = preproc_intensity(weights_mat)
                print('Preprocessed intensity')

            kesten_bin_avg, _, kesten_bin_diff_avg, _, kesten_bin_diff_var = bin_equal_addmult(
                weights_mat,
                compute_diff=fit_to_difference,
                exclude_zero=True, # we set this to True, just to fit to weights that are present
                use_logs=False,
                bin_size=bin_size,
                abs_val=False,
                sq_within_bin=False,
                ret_bins=False
            )

        self.kesten_bin_avg = kesten_bin_avg
        self.kesten_bin_diff_avg = kesten_bin_diff_avg
        self.kesten_bin_diff_var = kesten_bin_diff_var
        self.kesten_bin_diff_std = np.sqrt(self.kesten_bin_diff_var)

        self.kesten_var_x = np.square(self.kesten_bin_avg)

        self.kesten_mean_reg, self.kesten_mean_reg_train_idxs, self.kesten_mean_reg_test_idxs = linear_regress(X=self.kesten_bin_avg,
                                      Y=self.kesten_bin_diff_avg,
                                      return_idxs=True,
                                      train_frac=train_frac,
                                      fit_intercept=fit_intercept)
        self.kesten_var_reg, self.kesten_var_reg_train_idxs, self.kesten_var_reg_test_idxs = linear_regress(X=self.kesten_var_x,
                                     Y=self.kesten_bin_diff_var,
                                     return_idxs=True,
                                     train_frac=train_frac,
                                     fit_intercept=fit_intercept)
        self.kesten_std_reg, self.kesten_std_reg_train_idxs, self.kesten_std_reg_test_idxs = linear_regress(X=self.kesten_bin_avg,
                                     Y=self.kesten_bin_diff_std,
                                     return_idxs=True,
                                     train_frac=train_frac,
                                     fit_intercept=fit_intercept)

        assert(np.array_equal(self.kesten_mean_reg_test_idxs, self.kesten_var_reg_test_idxs))
        self.kesten_eval_idxs = self.kesten_mean_reg_test_idxs if train_frac < 1.0 else self.kesten_mean_reg_train_idxs
        self.kesten_mean_X_test = np.expand_dims(self.kesten_bin_avg[self.kesten_eval_idxs], axis=-1)
        self.kesten_mean_Y_test = self.kesten_bin_diff_avg[self.kesten_eval_idxs]
        self.kesten_var_X_test = np.expand_dims(self.kesten_var_x[self.kesten_eval_idxs], axis=-1)
        self.kesten_var_Y_test = self.kesten_bin_diff_var[self.kesten_eval_idxs]
        self.kesten_std_X_test = np.expand_dims(self.kesten_bin_avg[self.kesten_eval_idxs], axis=-1)
        self.kesten_std_Y_test = self.kesten_bin_diff_std[self.kesten_eval_idxs]
        self.kesten_mean_test_rsquared = self.kesten_mean_reg.score(self.kesten_mean_X_test, self.kesten_mean_Y_test)
        self.kesten_var_test_rsquared = self.kesten_var_reg.score(self.kesten_var_X_test, self.kesten_var_Y_test)
        self.kesten_std_test_rsquared = self.kesten_std_reg.score(self.kesten_std_X_test, self.kesten_std_Y_test)
        self.kesten_a = self.kesten_mean_reg.coef_[0]
        self.kesten_b = self.kesten_var_reg.coef_[0]
        self.kesten_c = self.kesten_mean_reg.intercept_
        self.kesten_d = self.kesten_var_reg.intercept_
        self.kesten_mean_line = lambda x: self.kesten_a * x + self.kesten_c
        self.kesten_mean_line_test = self.kesten_mean_line(self.kesten_mean_X_test)
        self.kesten_var_line = lambda x: self.kesten_b * x + self.kesten_d
        self.kesten_var_line_test = self.kesten_var_line(self.kesten_var_X_test)
        self.kesten_std_line = lambda x: self.kesten_std_reg.coef_[0] * x + self.kesten_std_reg.intercept_
        self.kesten_std_line_test = self.kesten_std_line(self.kesten_std_X_test)

    def simulate_kesten(self,
                        **simulation_kwargs):
        """simulate_kesten process.
        """
        # you need to fit the kesten first

        self.kesten_weight_matrix = run_kesten(a=self.kesten_a,
                                               b=self.kesten_b,
                                               c=self.kesten_c,
                                               d=self.kesten_d,
                                               **simulation_kwargs)

    def plot_conditional_mat(
        self,
        fig=None,
        ax=None,
        standalone_fig=True,
        include_colorbar=True,
        scale_factor=1.0,
        cmap=None,
        save_nm=None,
        fig_kwargs={},
    ):
        """plot_conditional_mat.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        include_colorbar :
            include_colorbar
        cmap :
            cmap
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """

        assert self.markov_out is not None
        if standalone_fig:
            plt.clf()
        if ax is None:
            plt.imshow(self.markov_out["conditional_dist"], cmap=cmap)
        else:
            im = ax.imshow(self.markov_out["conditional_dist"], cmap=cmap)
        plt.ylabel("Day $T$ State", size=40*scale_factor, weight="bold")
        plt.xlabel("Day $T+4$ State", size=40*scale_factor, weight="bold")
        plt.xticks(size=25*scale_factor)
        plt.yticks(size=25*scale_factor)
        if include_colorbar:
            if ax is None:
                plt.colorbar()
            else:
                assert(fig is not None)
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                axins = inset_axes(ax,
                                   width="5%",  # width = 5% of parent_bbox width
                                   height="100%",  # height : 50%
                                   loc='lower left',
                                   bbox_to_anchor=(1.05, 0., 1, 1),
                                   bbox_transform=ax.transAxes,
                                   borderpad=0,
                                   )
                cbar = fig.colorbar(im, cax=axins,
                                    ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                cbar.ax.tick_params(labelsize=20*scale_factor)

        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def overlay_kesten_markov_states(self, num_points=1000, exclude_last_bin=True):
        def kesten_to_cont_state(cont_val, state_tuples):
            found_cont_state = None
            for state_idx, state_rng in enumerate(state_tuples):
                if (state_rng[0] <= cont_val) and (cont_val < state_rng[1]):
                    interval_frac = (cont_val - state_rng[0]) / (state_rng[1] - state_rng[0])
                    found_cont_state = state_idx + interval_frac
                    break
            return found_cont_state

        self.fit_kesten_procedure(fit_to_difference=False)
        state_tuples = list(self.markov_out['stationary_dist'].keys())
        x = [state_idx for state_idx, state_rng in enumerate(state_tuples)]
        if exclude_last_bin:
            x_cont = np.linspace(start=x[0], stop=x[-1], num=num_points, endpoint=True)
        else:
            x_cont = np.linspace(start=x[0], stop=x[-1]+1, num=num_points, endpoint=False)
        x_cont_state = np.floor(x_cont)
        mean_y = []
        mean_y_above_1std = []
        mean_y_below_1std = []
        for curr_x_idx, curr_x in enumerate(x_cont):
            curr_x_state = x_cont_state[curr_x_idx]
            curr_frac_interval = curr_x - curr_x_state
            curr_state_rng = state_tuples[(int)(curr_x_state)]
            # map to the raw weight values to get kesten values
            curr_raw_x = curr_state_rng[0] + curr_frac_interval*(curr_state_rng[1] - curr_state_rng[0])
            curr_raw_mean = self.kesten_mean_line(curr_raw_x)
            curr_raw_std = self.kesten_std_line(curr_raw_x)
            curr_raw_above_1std = curr_raw_mean + curr_raw_std
            curr_raw_below_1std = curr_raw_mean - curr_raw_std
            # map kesten values back to state indices
            mean_y.append(kesten_to_cont_state(curr_raw_mean, state_tuples))
            mean_y_above_1std.append(kesten_to_cont_state(curr_raw_above_1std, state_tuples))
            mean_y_below_1std.append(kesten_to_cont_state(curr_raw_below_1std, state_tuples))

        self.kesten_x_cont = x_cont
        self.kesten_mean_y = mean_y
        self.kesten_mean_y_above_1std = mean_y_above_1std
        self.kesten_mean_y_below_1std = mean_y_below_1std

    def compute_survival_fraction(self,
                                  num_time,
                                  weights_matrix,
                                  syn_idx_to_track):

        if isinstance(num_time, int):
            num_time = range(num_time)

        num_start_synapses = 1.0*len(syn_idx_to_track)
        survival_frac = [np.count_nonzero(weights_matrix[syn_idx_to_track, t])/(num_start_synapses) for t in num_time]
        return survival_frac

    def plot_survival_fraction(
        self,
        ax=None,
        standalone_fig=True,
        num_time=None,
        num_seeds=1,
        start_idx=0,
        lw=10.0,
        start_distribution=None,
        exclude_zero_at_start=True,
        make_zero_absorbing=True,
        dendrite_map=None,
        lbl_nm=None,
        from_data=False,
        save_nm=None,
        fig_kwargs={},
        return_mat=False
    ):
        """plot_weights_addmult.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """

        if standalone_fig:
            plt.clf()
        if ax is None:
            ax = plt.subplot(111)

        markov_seeds = []
        if from_data:
            num_seeds = 1
            for i in range(num_seeds):
                markov_seeds.append(self.weights)
        else:
            for i in range(num_seeds):
                self.simulate_markov_chain(start_distribution=start_distribution,
                                           exclude_zero_at_start=exclude_zero_at_start,
                                           make_zero_absorbing=make_zero_absorbing,
                                           seed_val=i)
                markov_seeds.append(self.chain_state_matrix)

        if num_time is None:
            num_time = markov_seeds[0].shape[1]
        else:
            if isinstance(num_time, int):
                if start_idx is None:
                    start_idx = 0
                num_time = np.arange(start_idx, num_time)
            assert(num_time[-1] <= markov_seeds[0].shape[1])


        survival_frac_mat = []
        for i in range(num_seeds):
            curr_seed_weights = markov_seeds[i]
            curr_seed_synapses_to_track = np.where(curr_seed_weights[:, 0] != 0.0)[0]
            if dendrite_map is not None:
                assert(dendrite_map.shape[0] == curr_seed_weights.shape[0])
                dendrites_to_track = np.unique(dendrite_map[curr_seed_synapses_to_track])
                for d in dendrites_to_track:
                    # compute survival fraction for all synapses we want to track that are on that dendrite
                    curr_dendrite_syn_idx = np.where(dendrite_map == d)[0]
                    curr_seed_curr_dendrite_synapses_to_track = list(set(curr_dendrite_syn_idx) & set(curr_seed_synapses_to_track))
                    curr_seed_curr_dendrite_survival_frac = self.compute_survival_fraction(num_time=num_time,
                                                                                           weights_matrix=curr_seed_weights,
                                                                                           syn_idx_to_track=curr_seed_curr_dendrite_synapses_to_track)
                    survival_frac_mat.append(curr_seed_curr_dendrite_survival_frac)
            else:
                curr_seed_survival_frac = self.compute_survival_fraction(num_time=num_time,
                                                                         weights_matrix=curr_seed_weights,
                                                                         syn_idx_to_track=curr_seed_synapses_to_track)
                survival_frac_mat.append(curr_seed_survival_frac)

        survival_frac_mat = np.array(survival_frac_mat)

        if self.cell_type == 'pv':
            color = 'r'
        else:
            color = 'b'

        if from_data:
            plt.fill_between(x=num_time, y1=np.mean(survival_frac_mat, axis=0)-sem(survival_frac_mat, axis=0),
                         y2=np.mean(survival_frac_mat, axis=0)+sem(survival_frac_mat, axis=0),
                         linewidth=lw,
                         color=color,
                         alpha=0.1,
                         label=self.cell_type if lbl_nm is None else lbl_nm)
        else:
            plt.errorbar(x=num_time, y=np.mean(survival_frac_mat, axis=0),
                         yerr=sem(survival_frac_mat, axis=0),
                         linewidth=lw,
                         capsize=10,
                         markersize=10,
                         marker='o',
                         color=color,
                         label=self.cell_type if lbl_nm is None else lbl_nm)

        plt.legend(loc='lower left', prop={"size": 40, "weight": "bold"})
        plt.xticks(np.arange(8), 4*np.arange(8), fontsize=50, fontweight='bold')
        plt.yticks(fontsize=50, fontweight='bold')
        plt.xlabel('Days', fontsize=50, fontweight="bold")
        plt.ylabel('Survival Fraction', fontsize=50, fontweight="bold")
        plt.ylim([0.0, 1.05])
        plt.xlim([start_idx, num_time[-1]])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()
        if return_mat:
            return survival_frac_mat

    def plot_conditional_std_unbinned(self,
                                      ax=None,
                                      standalone_fig=True,
                                      save_nm=None,
                                      exclude_zero_weight=True,
                                      use_bootstrap=True,
                                      fig_kwargs={}):

        if standalone_fig:
            plt.clf()
        if ax is None:
            ax = plt.subplot(111)

        if use_bootstrap:
            assert self.bootstrap_markov_out is not None

            bootstrap_runs = self.bootstrap_markov_out['stationary_dist'].shape[0]
            bootstrap_conditional_std_unbinned = []
            for i in range(bootstrap_runs):
                conditional_std_unbinned = compute_conditional_std_unbinned(intensity_mat=self.bootstrap_markov_out['intensity_mat'][i],
                                                                            stationary_dist=self.bootstrap_markov_out['stationary_dist_dict'][i],
                                                                            exclude_zero_weight=exclude_zero_weight,
                                                                            ret_array=True)

                bootstrap_conditional_std_unbinned.append(np.expand_dims(conditional_std_unbinned, axis=0))

            bootstrap_conditional_std_unbinned = np.concatenate(bootstrap_conditional_std_unbinned, axis=0)

            bootstrap_conditional_std_unbinned_mean, bootstrap_conditional_std_unbinned_std = compute_bootstrap_mean_std(
                bootstrap_conditional_std_unbinned
            )
            num_states = len(bootstrap_conditional_std_unbinned_mean)
            ax.bar(range(num_states), bootstrap_conditional_std_unbinned_mean, yerr=bootstrap_conditional_std_unbinned_std, capsize=5)
        else:
            conditional_std_unbinned = compute_conditional_std_unbinned(intensity_mat=self.weights,
                                                                        stationary_dist=self.markov_out['stationary_dist'],
                                                                        exclude_zero_weight=exclude_zero_weight,
                                                                        ret_array=True)
            num_states = len(conditional_std_unbinned)
            plt.bar(range(num_states), conditional_std_unbinned)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xticks(range(num_states))
        plt.xlabel("State (${s}_{T}$) i", fontsize=25, fontweight="bold")
        plt.ylabel(
            "$\sigma$", fontsize=25, fontweight="bold"
        )
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def plot_stationary_vs_empirical(
        self,
        ax=None,
        standalone_fig=True,
        color_1="b",
        color_2="orange",
        width=1.0,
        save_nm=None,
        fig_kwargs={},
        xtick_ub=27.5,
        xtick_factor=1,
        scale_factor=1.0
    ):
        """plot_stationary_vs_empirical.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        color_1 :
            color_1
        color_2 :
            color_2
        width :
            width
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """

        assert self.bootstrap_markov_out is not None
        if standalone_fig:
            plt.clf()

        stationary_emp_mean, stationary_emp_std = compute_bootstrap_mean_std(
            self.bootstrap_markov_out["stationary_dist"]
        )
        stationary_model_mean, stationary_model_std = compute_bootstrap_mean_std(
            self.bootstrap_markov_out["analytic_stationary_dist"]
        )
        bincounts_emp_mean, bincounts_emp_std = compute_bootstrap_mean_std(
            self.bootstrap_markov_out["bin_counts_mat"]
        )

        if ax is None:
            ax = plt.subplot(111)
        idx_k = 0
        for idx, v in enumerate(list(stationary_emp_mean)):
            if idx == 0:
                label_1 = "Data"
                label_2 = "Model"
            else:
                label_1 = None
                label_2 = None
            ax.bar(
                idx_k - 0.5,
                v,
                color=color_1,
                width=width,
                label=label_1,
                yerr=stationary_emp_std[idx],
                capsize=5
            )
            ax.bar(
                idx_k + 0.5,
                stationary_model_mean[idx],
                color=color_2,
                width=width,
                label=label_2,
                yerr=stationary_model_std[idx],
                capsize=5
            )
            idx_k += 2.5
        plt.xticks(np.arange(0, xtick_ub, 2.5*xtick_factor), np.arange(0, len(stationary_emp_mean), xtick_factor),
                   fontsize=25*scale_factor)
        plt.xlabel('State', fontsize=40, fontweight='bold')
        plt.yticks(fontsize=25*scale_factor)
        plt.ylabel('Stationary Distribution', fontsize=40*scale_factor, fontweight='bold')
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.legend(loc="best", prop={'weight':'bold', 'size':25*scale_factor})
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def bootstrap_detailed_balance(self):
        """bootstrap_detailed_balance.
        """
        p_ijp_j_agg = []
        p_jip_i_agg = []
        for k in range(self.bootstrap_markov_out["conditional_dist"].shape[0]):
            p_ijp_j, p_jip_i = verify_detailed_balance(
                self.bootstrap_markov_out["conditional_dist"][k].T,
                self.bootstrap_markov_out["analytic_stationary_dist"][k],
            )
            p_ijp_j_agg.append(np.expand_dims(p_ijp_j, axis=0))
            p_jip_i_agg.append(np.expand_dims(p_jip_i, axis=0))
        p_ijp_j_agg = np.concatenate(p_ijp_j_agg, axis=0)
        p_jip_i_agg = np.concatenate(p_jip_i_agg, axis=0)

        mean_p_ijp_j_agg, std_p_ijp_j_agg = compute_bootstrap_mean_std(p_ijp_j_agg)
        mean_p_jip_i_agg, std_p_jip_i_agg = compute_bootstrap_mean_std(p_jip_i_agg)
        return mean_p_ijp_j_agg, std_p_ijp_j_agg, mean_p_jip_i_agg, std_p_jip_i_agg

    def plot_detailed_balance(
        self,
        ax=None,
        standalone_fig=True,
        diagonal_start=[-0.005, 0.01],
        diagonal_end=[-0.005, 0.01],
        xlim=[-0.0001, 0.003],
        ylim=[-0.001, 0.006],
        save_nm=None,
        fig_kwargs={},
    ):
        """plot_detailed_balance.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        diagonal_start :
            diagonal_start
        diagonal_end :
            diagonal_end
        xlim :
            xlim
        ylim :
            ylim
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """

        if standalone_fig:
            plt.clf()
        (
            mean_p_ijp_j_agg,
            std_p_ijp_j_agg,
            mean_p_jip_i_agg,
            std_p_jip_i_agg,
        ) = self.bootstrap_detailed_balance()
        print(pearsonr(mean_p_ijp_j_agg, mean_p_jip_i_agg)[0])
        combined_error = np.sqrt(
            np.square(std_p_ijp_j_agg) + np.square(std_p_jip_i_agg)
        )
        if ax is None:
            ax = plt.subplot(111)
        plt.errorbar(mean_p_ijp_j_agg, mean_p_jip_i_agg, yerr=combined_error, fmt="o")
        if (diagonal_start is not None) and (diagonal_end is not None):
            plt.plot(diagonal_start, diagonal_end, color="k")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("$p(j\mid i)\pi(i)$", fontsize=25, fontweight="bold")
        plt.ylabel("$p(i\mid j)\pi(j)$", fontsize=25, fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def plot_weights_addmult(
        self,
        ax=None,
        standalone_fig=True,
        exclude_zero=False,
        use_logs=False,
        bin_size=14,
        save_nm=None,
        fig_kwargs={},
    ):
        """plot_weights_addmult.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        exclude_zero :
            exclude_zero
        use_logs :
            use_logs
        bin_size :
            bin_size
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """

        if standalone_fig:
            plt.clf()
        if ax is None:
            ax = plt.subplot(111)
        bin_avg, bin_sem, bin_diff_avg, bin_diff_sem = bin_equal_addmult(
            self.weights,
            exclude_zero=exclude_zero,
            use_logs=use_logs,
            bin_size=bin_size,
        )

        combined_sem = np.sqrt(np.square(bin_sem) + np.square(bin_diff_sem))
        slope, intercept, r_value, p_value, std_err = linregress(bin_avg, bin_diff_avg)
        print("Slope", slope, "r val", r_value, "p val", p_value, "std err", std_err)
        plt.errorbar(
            bin_avg, bin_diff_avg, color="b", marker="o", yerr=combined_sem, ls="none"
        )
        plt.plot(bin_avg, slope * bin_avg + intercept, linewidth=4.0, color="r")
        plt.xlabel("$|w(T)|$", fontsize=25, fontweight="bold")
        plt.ylabel("$|w(T+4) - w(T)|$", fontsize=25, fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def bootstrap_conditional_mean_std(self):
        """bootstrap_conditional_mean_var.
        """
        mean_agg = []
        var_agg = []
        for i in range(self.bootstrap_markov_out["conditional_dist"].shape[0]):
            conditional_mean, conditional_variance = compute_conditional_mean_var(
                self.bootstrap_markov_out["conditional_dist"][i], ret_array=True
            )
            var_agg.append(np.expand_dims(conditional_variance, axis=0))
            mean_agg.append(np.expand_dims(conditional_mean, axis=0))

        var_agg = np.concatenate(var_agg, axis=0)
        std_agg = np.sqrt(var_agg)
        mean_std_agg, std_std_agg = compute_bootstrap_mean_std(std_agg)
        mean_agg = np.concatenate(mean_agg, axis=0)
        mean_mean_agg, std_mean_agg = compute_bootstrap_mean_std(mean_agg)
        return mean_mean_agg, std_mean_agg, mean_std_agg, std_std_agg

    def bootstrap_conditional_mean_var(self):
        """bootstrap_conditional_mean_var.
        """
        mean_agg = []
        var_agg = []
        for i in range(self.bootstrap_markov_out["conditional_dist"].shape[0]):
            conditional_mean, conditional_variance = compute_conditional_mean_var(
                self.bootstrap_markov_out["conditional_dist"][i], ret_array=True
            )
            var_agg.append(np.expand_dims(conditional_variance, axis=0))
            mean_agg.append(np.expand_dims(conditional_mean, axis=0))

        var_agg = np.concatenate(var_agg, axis=0)
        std_agg = var_agg # JBM Workaround to get var not
        mean_std_agg, std_std_agg = compute_bootstrap_mean_std(std_agg)
        mean_agg = np.concatenate(mean_agg, axis=0)
        mean_mean_agg, std_mean_agg = compute_bootstrap_mean_std(mean_agg)
        return mean_mean_agg, std_mean_agg, mean_std_agg, std_std_agg

    def plot_conditional_mean(
        self, ax=None, standalone_fig=True, save_nm=None, fig_kwargs={}
    ):
        """plot_conditional_mean.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """
        if standalone_fig:
            plt.clf()
        num_states = self.bootstrap_markov_out["stationary_dist"].shape[1]
        mean_mean_agg, std_mean_agg, _, _ = self.bootstrap_conditional_mean_var()
        slope, intercept, r_value, p_value, std_err = linregress(
            range(num_states), mean_mean_agg
        )
        print("Slope", slope, "r val", r_value, "p val", p_value, "std err", std_err)
        if ax is None:
            ax = plt.subplot(111)
        xi = np.arange(num_states)
        plt.errorbar(
            xi, mean_mean_agg, marker="o", yerr=std_mean_agg, ls="none", color="b"
        )
        # plt.plot(xi, xi * slope + intercept, linewidth=4.0, color="r")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xticks(range(num_states))
        plt.xlabel("State (${s}_{T}$) i", fontsize=25, fontweight="bold")
        plt.ylabel("$E[{s}_{T+4}\mid {s}_{T} = i]$", fontsize=25, fontweight="bold")
        plt.plot(np.arange(num_states))
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def plot_conditional_var(
        self, ax=None, standalone_fig=True, save_nm=None, fig_kwargs={}, set_yticks=True
    ):
        """plot_conditional_std.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """

        if standalone_fig:
            plt.clf()
        num_states = self.bootstrap_markov_out["stationary_dist"].shape[1]
        _, _, mean_std_agg, std_std_agg = self.bootstrap_conditional_mean_var()
        _, _, mean_std_agg, std_std_agg = self.bootstrap_conditional_mean_var()
        slope, intercept, r_value, p_value, std_err = linregress(range(num_states), mean_std_agg)

        print("Slope", slope, "r val", r_value, "p val", p_value, "std err", std_err)

        if ax is None:
            ax = plt.subplot(111)
        xi = np.arange(num_states)
        plt.errorbar(
            xi, mean_std_agg, marker="o", yerr=std_std_agg, ls="none", color="b"
        )
        plt.plot(xi, xi * slope + intercept, linewidth=4.0, color="r")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if set_yticks:
            plt.yticks([0.4, 1.2, 1.8], [0.4, 1.2, 1.8])
            plt.ylim(0,3)
        plt.xticks(range(num_states))
        plt.xlabel("State (${s}_{T}$) i", fontsize=25, fontweight="bold")
        plt.ylabel(
            "$\sigma^2$", fontsize=25, fontweight="bold"
        )
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def plot_conditional_std(
        self, ax=None, standalone_fig=True, save_nm=None, fig_kwargs={}, set_yticks=True
    ):
        """plot_conditional_std.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """

        if standalone_fig:
            plt.clf()
        num_states = self.bootstrap_markov_out["stationary_dist"].shape[1]
        _, _, mean_std_agg, std_std_agg = self.bootstrap_conditional_mean_std()
        _, _, mean_std_agg, std_std_agg = self.bootstrap_conditional_mean_std()
        slope, intercept, r_value, p_value, std_err = linregress(range(num_states), mean_std_agg)

        print("Slope", slope, "r val", r_value, "p val", p_value, "std err", std_err)

        if ax is None:
            ax = plt.subplot(111)
        xi = np.arange(num_states)
        plt.errorbar(
            xi, mean_std_agg, marker="o", yerr=std_std_agg, ls="none", color="b"
        )
        plt.plot(xi, xi * slope + intercept, linewidth=4.0, color="r")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if set_yticks:
            plt.yticks([0.4, 1.2, 1.8], [0.4, 1.2, 1.8])
            plt.ylim(0,2)
        plt.xticks(range(num_states))
        plt.xlabel("State (${s}_{T}$) i", fontsize=25, fontweight="bold")
        plt.ylabel(
            "$\sigma({s}_{T+4}\mid {s}_{T} = i)$", fontsize=25, fontweight="bold"
        )
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def compute_crosscorr_seq(self,
                              weight_matrix,
                              x_rng=None,
                              peak_normalize=False):

        """compute_crosscorr_seq.

        Parameters
        ----------
        weight_matrix :
            weight_matrix
        x_rng :
            x_rng
        """
        if x_rng is None:
            # assumes weight matrix is a synpases x days matrx
            x_max = weight_matrix.shape[1] - 1
            x_min = -1.0 * (x_max - 1)
            x_rng = np.arange(x_min, x_max)

        crosscorr_seq = np.array([cross_correlation(delta=i, weight_matrix=weight_matrix) for i in x_rng])
        if peak_normalize:
            crosscorr_seq /= np.amax(crosscorr_seq)

        return crosscorr_seq

    def plot_timeconstant(
        self,
        ax=None,
        standalone_fig=True,
        x_rng=None,
        peak_normalize=False,
        lw=4.0,
        save_nm=None,
        fig_kwargs={},
    ):
        """plot_timeconstant.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        x_rng :
            x_rng
        lw :
            lw
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """

        if standalone_fig:
            plt.clf()
        if ax is None:
            ax = plt.subplot(111)
        if x_rng is None:
            x_max = self.weights.shape[1] - 1
            x_min = -1.0 * (x_max - 1)
            x_rng = np.arange(x_min, x_max)
        plt.plot(
            self.compute_crosscorr_seq(weight_matrix=self.weights, x_rng=x_rng, peak_normalize=peak_normalize),
            label="data",
            linewidth=lw,
        )
        # need to run simulate markov chain first!
        if self.chain_weight_matrix is not None:
            plt.plot(
                self.compute_crosscorr_seq(
                    weight_matrix=self.chain_weight_matrix, x_rng=x_rng, peak_normalize=peak_normalize
                ),
                label="chain",
                linewidth=lw,
            )
        plt.xticks(range(len(x_rng)), x_rng * 4, size=20, weight="bold")
        plt.yticks(size=20, weight="bold")
        plt.legend(loc="best", prop={"size": 20, "weight": "bold"})
        plt.xlabel("$\Delta$ (days)", size=20, weight="bold")
        plt.ylabel("$(w_s(t), w_s(t+\Delta))$", size=20, weight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def plot_velocity(
        self,
        ax=None,
        standalone_fig=True,
        x_rng=None,
        peak_normalize=False,
        lw=4.0,
        scale_factor=1.0,
        save_nm=None,
        fig_kwargs={},
    ):
        """plot_velocity.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        x_rng :
            x_rng
        lw :
            lw
        save_nm :
            save_nm
        fig_kwargs :
            fig_kwargs
        """

        if standalone_fig:
            plt.clf()
        if ax is None:
            ax = plt.subplot(111)
        data_velocity = compute_velocity(self.weights)
        if x_rng is None:
            x_max = self.weights.shape[1] - 1
            x_min = -1.0 * (x_max - 1)
            x_rng = np.arange(x_min, x_max)
        plt.plot(
            self.compute_crosscorr_seq(weight_matrix=data_velocity, x_rng=x_rng, peak_normalize=peak_normalize),
            label="Data",
            linewidth=lw
        )
        # need to run simulate markov chain first!
        if self.chain_weight_matrix is not None:
            chain_velocity = compute_velocity(self.chain_weight_matrix)
            plt.plot(
                self.compute_crosscorr_seq(weight_matrix=chain_velocity, x_rng=x_rng, peak_normalize=peak_normalize),
                label="Model",
                linewidth=lw,
            )
        plt.xticks(range(len(x_rng)), x_rng * 4, size=20*scale_factor, weight="bold")
        plt.yticks(size=25*scale_factor, weight="bold")
        plt.legend(loc="best", prop={"size": 25*scale_factor, "weight": "bold"})
        plt.xlabel("$\Delta$ (days)", size=40*scale_factor, weight="bold")
        plt.ylabel("$(v_s(t), v_s(t+\Delta))$", size=40*scale_factor, weight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if save_nm is not None:
            save_figure(save_nm, user=self.user, **fig_kwargs)
        if standalone_fig:
            plt.show()

    def compute_bta_dta(self,
                        weights_mat=None,
                        preproc_intensity_mat=True,
                        average_over_missing=False,
                        use_all_synapses=False):
        """compute_bta_dta.

        Parameters
        ----------
        return_sem :
            return_sem
        use_all_synapses :
            use_all_synapses
        """

        if weights_mat is None:
            weights_mat = self.weights
        if preproc_intensity_mat:
            weights_mat = preproc_intensity(weights_mat)

        if use_all_synapses:
            bta_relevant_syn_idx = None
            dta_relevant_syn_idx = None
        else:
            dynamics_mat = create_dynamic_events_matrix(weights_mat)
            # use only added synapses for bta and eliminated synapses for dta
            bta_relevant_syn_idx = np.where(dynamics_mat == 1)[0]
            dta_relevant_syn_idx = np.where(dynamics_mat == -1)[0]

        bta, bta_sem = weight_triggered_average(
            weights_mat,
            preproc_intensity_mat=preproc_intensity_mat,
            align_type="birth",
            average_over_missing=average_over_missing,
            return_aligned_mat=False,
            relevant_syn_idx=bta_relevant_syn_idx,
        )

        dta, dta_sem = weight_triggered_average(
            weights_mat,
            preproc_intensity_mat=preproc_intensity_mat,
            align_type="death",
            average_over_missing=average_over_missing,
            return_aligned_mat=False,
            relevant_syn_idx=dta_relevant_syn_idx,
        )
        return bta, bta_sem, dta, dta_sem

    def plot_bta_dta(
        self,
        ax=None,
        standalone_fig=True,
        weights_mat=None,
        preproc_intensity_mat=True,
        average_over_missing=False,
        bta_cutoff_idx=5,
        include_corr=False,
        use_all_synapses=False,
        mode="all",
        label=None,
        color=None
    ):
        """plot_bta_dta.

        Parameters
        ----------
        ax :
            ax
        standalone_fig :
            standalone_fig
        bta_cutoff_idx :
            bta_cutoff_idx
        include_corr :
            include_corr
        use_all_synapses :
            use_all_synapses
        """

        if weights_mat is None:
            weights_mat = self.weights

        if bta_cutoff_idx is None:
            bta_cutoff_idx = weights_mat.shape[1]
        assert bta_cutoff_idx <= weights_mat.shape[1]

        bta, bta_sem, dta, dta_sem = self.compute_bta_dta(
            weights_mat=weights_mat,
            preproc_intensity_mat=preproc_intensity_mat,
            average_over_missing=average_over_missing,
            use_all_synapses=use_all_synapses
        )
        if standalone_fig:
            plt.clf()
        if ax is None:
            ax = plt.subplot(111)
        dta_start_idx = weights_mat.shape[1] - bta_cutoff_idx
        x_rng = np.arange(bta_cutoff_idx)
        bta_toplot = bta[:bta_cutoff_idx]
        dta_toplot = dta[dta_start_idx:][::-1]
        if mode.lower() == "birth":
            plt.errorbar(
                x_rng,
                bta_toplot,
                label="Birth Trajectory" if label is None else label,
                linewidth=4.0,
                yerr=bta_sem[:bta_cutoff_idx],
                color="b" if color is None else color,
                capsize=5
            )
        elif mode.lower() == "death":
            plt.errorbar(
                x_rng,
                dta_toplot,
                label="Death Trajectory" if label is None else label,
                linewidth=4.0,
                yerr=dta_sem[dta_start_idx:][::-1],
                color="orange"  if color is None else color,
                capsize=5
            )
        else:
            plt.errorbar(
                x_rng,
                bta_toplot,
                label="Birth Trajectory",
                linewidth=4.0,
                yerr=bta_sem[:bta_cutoff_idx],
                color="b",
                capsize=5
            )
            plt.errorbar(
                x_rng,
                dta_toplot,
                label="Death Trajectory",
                linewidth=4.0,
                yerr=dta_sem[dta_start_idx:][::-1],
                color="orange",
                capsize=5
            )
        plt.xticks(x_rng, 4 * x_rng)
        plt.xlabel("Observation Days", fontweight="bold")
        plt.ylabel("Synaptic Strength", fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.legend(loc="upper left", prop={"weight": "bold"})
        if include_corr:
            plt.title("Corr: {}".format(pearsonr(bta_toplot, dta_toplot)[0]))
        if standalone_fig:
            plt.show()
