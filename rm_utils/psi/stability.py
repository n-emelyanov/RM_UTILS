import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pickle

from math import isinf
import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)


class StabilityIndexCalculator:
	"""Calculating PSI coefficients
	Two main methods:
	calculate - used on pd.DataFrame with variable and time/group column.
	fit/predict - used to fit on one variable and to calculate on other variables.

	Saving/loading methods.
	load_fit
	save_fit

	Additional methods for output format. Recommended to use after calculate.
	- make_wide_table (rows-variables, columns-periods)
	- make_stat_summary_table (rows-variables+bins, columns-stats)
	- get_psi_table (single fit/predict shortcut)
	- plot_calculations

	PSI
	If user not presented custom bins then:
	Quantile binning for expected values (initial variable), then using same bins for observed varaibles.

	Init params
	------------
	min_bin_coeff: float [0-1]
	Minimum bin size for quantile cut. Share of expected bin size = len(variable) / n_bins)

	min_bin_size_cat: float [0-1]
		Minimum relative bin size for category
	"""

	def __init__(self, min_bin_coeff=0.3, min_bin_size_cat=0.03):
		self.na_str = "_missing"
		self.out_int_str = "_out_of_interval"
		self.min_bin_coeff = min_bin_coeff
		self.initial_val = "expected"
		self.psi_str = "psi"
		# fitdata
		self.fit_data = {}
		self.fit_data["bins_dict"] = {}
		self.min_bin_size_cat = min_bin_size_cat

		# history variable to return fit/predict bin counts
		self.calc_bins_counts = {}
		self.calc_bins_nobs = {}
		self.left_minv = 0.0001
		self.min_left_std = 0.0001

	def calculate(
		self,
		df: pd.DataFrame,
		var_names: list,
		group_col: str,
		fit=True,
		initial_val=None,
		fit_mask=None,
		n_bins=10,
		exclude_miss=False,
		exclude_out_int=False,
		bin_edge_std=None,
		variable_bins={},
		variable_n_bins={},
		return_bin_counts=False,
		verbose=False,
		from_sql=None,  # Experimental TODO
		distrib_targ=None,
	):
		"""
		Calculate PSI for a variable on pandas dataframe.
		Note - first group also calculated and will have PSI = 0.
		Returns dict (key=variable) with PD dataframes containing PSI value per group.
		If you need values per bins, use - return_bin_counts.

		Parameters
		----------
		df: pd.DataFrame
			Data

		var_names: list
			Names of columns in dataframe

		group_col: str
			Column on which to perform splitting the data.

		fit: bool, default True
			If already fitted, use False.

		initial_val: scalar
			If group_col is not a datetime dtype, this value from group_col will be taken for initial sample.

		fit_mask: pd.Series, np.array
			Boolean mask for fit. Alternative for initial_val

		n_bins: int
			Number of bins. For categorical - max number of categories.

		exclude_miss: bool, default False
			Not calculate on missing values

		exclude_out_int: bool, default False
			Not calculate on values that falls out of initial variable bins.

		bin_edge_std: tuple, default (1, 1)
			Expand edge bins (min/max) values of initial variable. If = 0, then initiabl min/max values are used.
			Expanding (+-) on initial bins std value multiplied by this parameter.

		variable_bins: dict of array_like objects
			Fixed bins for variables, if need to calculate on user defined intervals. For example when monitoring WOE model.

		variable_n_bins: dict of ints
			Custom N bins for specific variable

		return_bin_counts: bool, default False
			Wheather to return counts per bin in each group. Then returns second variable.

		distrib_targ: str, default None
			Not PSI mode.
			Name of target column to use, instead of counting observations.
			Special parameter to calculate not normalised distribution of observation counts over some binned variable bins,
			but to calculate normalised mean of some other target variable over this bins and normalization happens by expected variable sum.
			This could answer questions like - Is the mean target of bins of this variable changed?
		"""

		group_col_counts = self._calc_group_col_counts(df, group_col, from_sql)

		if group_col_counts["counts"].min() < 100:
			logger.info("[INFO] Found groups with less then 100 observations")

		if not fit:
			expected_len = self.fit_data["expected_len"]
			self.initial_val = self.fit_data["initial_val"]
		else:
			if (initial_val is None) and (fit_mask is None):
				self.initial_val = group_col_counts[group_col].iloc[0]

				dtype_name = group_col_counts[group_col].dtype.name
				if ("date" in dtype_name) or ("period" in dtype_name):
					pass
				else:
					logger.info(f"[INFO] Not a datetime grouping. Initial category set as {self.initial_val}")
			else:
				self.initial_val = initial_val
			if fit_mask is None:
				expected_len = group_col_counts.loc[
					group_col_counts[group_col] == self.initial_val, "counts"
				].iloc[0]
			else:
				expected_len = group_col_counts["counts"].mean()

		if df[group_col].isna().any():
			logger.info(
				"[INFO] Found empty values in grouping column. Values will be ingnored"
			)

		result_stats = {}

		for _var in var_names:
			if verbose:
				print(f"Calculating {_var}")

			_fit_s = self._fit_calc_routine(df, fit, fit_mask, _var, group_col, expected_len, n_bins, variable_n_bins,
					from_sql, distrib_targ, exclude_miss, exclude_out_int, bin_edge_std, variable_bins)

			# check if fit were successfull
			if not (isinstance(_fit_s, str) and _fit_s == "error"):
				df_stats = self._stats_calc_routine(df, _var, group_col, distrib_targ)

				result_stats[_var] = df_stats

		if return_bin_counts:
			return result_stats, self.calc_bins_counts.copy()

		return result_stats

	def _calc_group_col_counts(self, df, group_col, from_sql):
		# define group col values
		if from_sql is not None:
			sql_table = from_sql["sql_table"]
			connection = from_sql["connection"]
			counts_script = f"SELECT {group_col}, COUNT(*) AS COUNTS FROM {sql_table} GROUP BY {group_col} ORDER BY {group_col}"
			group_col_counts = pd.read_sql(counts_script, con=connection)
		else:
			group_col_counts = (
				df[group_col]
				.value_counts()
				.sort_index()
				.reset_index()
				.rename(columns={"count": "counts"})
			)

		return group_col_counts

	def _fit_calc_routine(self, df, fit, fit_mask, _var, group_col, expected_len, n_bins, variable_n_bins,
			from_sql, distrib_targ, exclude_miss, exclude_out_int, bin_edge_std, variable_bins):
		if from_sql is not None:
			sql_table = from_sql["sql_table"]
			connection = from_sql["connection"]
			var_script = f"""SELECT {_var}, {group_col}
						FROM {sql_table}
						"""
			df = pd.read_sql(var_script, con=connection)

		if fit:
			# check n_bins
			_var_n_bins = variable_n_bins.get(_var, n_bins)
			if expected_len < 50 * _var_n_bins:
				logger.info(
					f"[INFO] Expected variable has too few values = {expected_len}"
				)
			elif expected_len < 10 * _var_n_bins:
				raise ValueError(
					f"Expected variable has too few values = {expected_len}. Reduce number of bins"
				)
			if fit_mask is None:
				fit_mask = df[group_col] == self.initial_val

			# not psi mode
			_fit_target = (
				None
				if distrib_targ is None
				else df.loc[fit_mask, distrib_targ]
			)

			_fit_s = self.fit(
				df.loc[fit_mask, _var],
				n_bins=_var_n_bins,
				exclude_miss=exclude_miss,
				exclude_out_int=exclude_out_int,
				bin_edge_std=bin_edge_std,
				bins=variable_bins.get(_var),
				targ_psi_ser=_fit_target,
			)
		else:
			_fit_s = "loaded"

		return _fit_s

	def _stats_calc_routine(self, df, _var, group_col, distrib_targ):
		# expected + observed
		grouped_data = df.groupby(group_col, observed=False)

		df_stats = pd.DataFrame()
		df_stats["n_obs"] = grouped_data[group_col].count()
		df_stats["n_nans"] = df_stats["n_obs"] - grouped_data[_var].count()

		df_stats["hitrate"] = (
			df_stats["n_obs"] - df_stats["n_nans"]
		) / df_stats["n_obs"]

		# var mean
		if pd.api.types.is_numeric_dtype(df[_var]):
			df_stats['var_mean'] = grouped_data[_var].mean()
		else:
			df_stats['var_mean'] = np.nan

		if distrib_targ is None:
			df_stats[self.psi_str] = grouped_data[_var].apply(
				lambda _df_var: self.predict(_df_var, var_name=_var)
			)
		else:
			# not psi mode. Need loop to rename series as group name. When groupby series this would be default
			assert (
				_var != distrib_targ
			), f"Target grouping column name same as calculate column = {_var}"
			for p_name, _df_group in grouped_data[[_var, distrib_targ]]:
				ser = _df_group[_var].copy()  # for renaming
				ser.name = p_name
				df_stats.loc[p_name, self.psi_str] = self.predict(
					ser, var_name=_var, targ_psi_ser=_df_group[distrib_targ]
				)

		return df_stats


	def fit(
		self,
		var_exp: pd.Series,
		n_bins=10,
		bins=None,
		exclude_miss=False,
		exclude_out_int=False,
		bin_edge_std=None,
		targ_psi_ser=None,
	):
		"""
		Fit given variable to compare with others during predict method.
		Make sure that variable has .name and during predict this fit_data will be taken by that name.

		Parameters
		----------
		var_exp : pd.Series
			Initial variable. The input array to be binned and compared. Must be 1-dimensional.
			Must have variable name in (.name) method.

		n_bins: int
			Number of bins. For categorical - max number of categories.

		bins: array-like
			Fixed bins for the variable, if need to calculate on user defined intervals. For example when monitoring WOE model.

		exclude_miss: bool, default False
			Not calculate on missing values

		exclude_out_int: bool, default False
			Not calculate on values that falls out of initial variable bins.

		bin_edge_std: tuple, default None equals to = (1, 1)
			Expand edge bins (min/max) values of initial variable. If = 0, then initiabl min/max values are used.
			Expanding (+-) on initial bins std value multiplied by this parameter.

		"""
		if var_exp.name is None:
			# raise ValueError("Expected variable has no name, won't be able to identify when predicting")
			logging.warning(
				"Expected variable has no name, won't be able to identify when predicting"
			)
		if pd.isnull(var_exp).all():
			logging.warning(
				f"[WARNING] Fit failed for {var_exp.name}. Expected variable is empty."
			)
			return "error"
			# raise ValueError(f"Expected variable is empty {var_exp.name}")

		if bin_edge_std is None:
			bin_edge_std = (1, 1)
		fit_data = {}
		self._vn = var_exp.name

		# saving fit params
		fit_data["low_unique"] = False
		fit_data["n_bins"] = n_bins
		fit_data["exclude_miss"] = exclude_miss
		fit_data["exclude_out_int"] = exclude_out_int
		fit_data["bin_edge_std"] = bin_edge_std
		fit_data["expected_len"] = len(var_exp)
		fit_data["initial_val"] = self.initial_val
		self.fit_data[self._vn] = fit_data.copy()
		self.fit_data["initial_val"] = self.initial_val
		self.fit_data["expected_len"] = len(var_exp)
		# binning and counting
		bin_var_exp, bins = self.bin_variable(
			var_exp, bins=bins, n_bins=n_bins
		)
		var_cnt_exp, base_counts = self._normalised_counts(bin_var_exp, targ_var=targ_psi_ser)
		# saving fit results
		self.fit_data[self._vn]["bins"] = bins
		# for use in calculate method
		self.fit_data["bins_dict"][self._vn] = bins
		self.fit_data[self._vn]["var_cnt_exp"] = var_cnt_exp  # counts for psi
		self.fit_data[self._vn]["var_nobs_exp"] = base_counts

		# updating counts/distribution table
		self._update_calc_counts_tab(self._vn, var_cnt_exp, base_counts, fit=True)
		return self

	def predict(
		self,
		x: pd.Series,
		return_table=False,
		var_name=None,
		per_name=None,
		targ_psi_ser=None,
	):
		"""
		Calculate total PSI value.
		Returns single value.

		Parameters
		----------
		x : array-like
		The input array to be binned and compared. Must be 1-dimensional.

		return_table: bool, default False
		Return PSI table with bins stats

		"""
		# when used in groupby x.name contains period name
		var_n = x.name if var_name is None else var_name
		self._vn = var_n
		period_n = x.name if per_name is None else per_name
		if self.fit_data.get(var_n) is None:
			logging.warning(f"{var_n} was not fitted. PSI predict failed")
			return None

		fit_data = self.fit_data[var_n]

		if pd.isnull(x).all():
			# if empty data, no binning required
			var_cnt_obs = pd.Series(
				data=np.zeros(len(fit_data["var_cnt_exp"])),
				name=period_n,
				index=fit_data["var_cnt_exp"].index,
			)
			var_cnt_obs.loc[self.na_str] = 1
			base_counts = var_cnt_obs.copy()
		else:
			# binning and counting
			bin_var_obs, bins = self.bin_variable(
				x, bins=fit_data["bins"], n_bins=fit_data["n_bins"]
			)
			var_cnt_obs, base_counts = self._normalised_counts(bin_var_obs, targ_var=targ_psi_ser)

		# make table and calculate index
		target_mode = targ_psi_ser is not None
		self.psi_tab = self._make_psi_table(
			fit_data["var_cnt_exp"], var_cnt_obs, target_mode, base_counts
		)

		psi_total = self._calculate_total_psi(
			self.psi_tab,
			exclude_miss=self.fit_data[self._vn]["exclude_miss"],
			exclude_out_int=self.fit_data[self._vn]["exclude_out_int"],
		)

		# updating counts/distribution table
		self._update_calc_counts_tab(var_n, var_cnt_obs, base_counts, fit=False)

		if return_table:
			return psi_total, self.psi_tab

		return psi_total

	############# saving / loading  ###################

	def load_fit(self, path):
		with open(path, "rb") as f:
			self.fit_data = pickle.load(f)

	def save_fit(self, path):
		with open(path, "wb") as f:
			pickle.dump(self.fit_data, f)

	############ USER OUTPUT UTILS #########################

	def make_wide_table(self, calc_tables_dict: dict, use_stat=None):
		"""
		Takes .calculate method result and combines into summary table
		"""
		if use_stat is None:
			use_stat = self.psi_str
		rows = []

		for _name, _table in calc_tables_dict.items():
			_row = _table[use_stat].T.copy()
			_row.name = _name
			rows.append(_row)
		df_res = pd.concat(rows, axis=1).T

		return df_res

	def make_stat_summary_table(self, calc_tables_dict: dict):
		"""Takes .calculate method result and combines into summary table"""
		rows = []

		for _name, _table in calc_tables_dict.items():
			if self.initial_val is None:
				psi_tab = _table.iloc[1:].copy()
			else:
				psi_tab = _table[_table.index != self.initial_val].copy()

			tab_info = dict(
				psi_min=psi_tab[self.psi_str].min(),
				psi_max=psi_tab[self.psi_str].max(),
				psi_mean=psi_tab[self.psi_str].mean(),
				hitrate_min=psi_tab["hitrate"].min(),
				hitrate_max=psi_tab["hitrate"].max(),
				hitrate_mean=psi_tab["hitrate"].mean(),
			)

			rows.append(pd.Series(tab_info, name=_name))
		df_summ = pd.concat(rows, axis=1)

		return df_summ

	def get_psi_table(
		self,
		df,
		var,
		group_var,
		fit_group,
		pred_group,
		n_bins=10,
		exclude_miss=False,
		exclude_out_int=False,
		bin_edge_std=None,
	):
		"""Shorcut for fit predict"""

		fit_mask = df[group_var] == fit_group
		pred_mask = df[group_var] == pred_group

		self.fit(
			df.loc[fit_mask, var],
			n_bins=n_bins,
			exclude_miss=exclude_miss,
			exclude_out_int=exclude_out_int,
			bin_edge_std=bin_edge_std,
		)

		psi, psi_tab = self.predict(df.loc[pred_mask, var], return_table=True)

		return psi_tab

	def beautify_bin_distr_table(self, psi_bin_counts, drop_bins=None, bin_symbols=3):
		"""
		Get wide table with each bin for each previously calculated variable

		Parameters
		----------
		drop_bins: array_like or None
		Dropping bins like 'out_of_interval' or '_missing'

		bin_symbols: int
		Output formatting. How many symbols keep in bins numbers
		"""
		drop_bins = [] if drop_bins is None else drop_bins
		c_tabs = psi_bin_counts

		if c_tabs is None:
			raise ValueError(
				"self.calc_counts_tabs is None. Check if calculate method were used beforehand"
			)

		rows = []

		for var, tab in c_tabs.items():
			# dropping bins and PSI bin columns
			tab = tab.loc[~tab.index.isin(drop_bins)]

			# beautify bins
			b_index = self._beautify_index_bins(tab.index, bin_symbols)

			# to multiindex
			index_tuples = list(zip([var] * len(tab.index), b_index))
			tab.index = pd.MultiIndex.from_tuples(
				index_tuples, names=["variable", "bins"]
			)

			rows.append(tab)
		df_res = pd.concat(rows)

		return df_res

	def plot_calculations(
		self,
		calc_tables_dict: dict,
		figsize=(12, 4),
		hlines=(0.1, 0.2),
		y_lim=(-0.05, 0.5),
		save_report_path=None,
	):
		"""Visualising result of calculate method"""
		# save report utils. Utility use conditional with clause
		class DummyWith:
			def __init__(self, *args, **kwargs):
				pass

			def __enter__(self):
				pass

			def __exit__(self, _type, value, traceback):
				pass

		to_save = save_report_path is not None
		with_cls = PdfPages if to_save else DummyWith

		with with_cls(f"{save_report_path}") as pdf:
			for _name, psi_tab in calc_tables_dict.items():
				_psi_tab = psi_tab.copy()
				_psi_tab.index = _psi_tab.index.astype(str)

				fig, ax = plt.subplots(figsize=figsize)
				ax2 = plt.twinx()

				ax.set_ylim(y_lim)
				ax.axhline(hlines[0], ls="-.", c="g")
				ax.axhline(hlines[1], ls="-.", c="r")
				ax.axhline(0, ls="-.", c="blue", lw=1, alpha=0.5)

				_psi_tab[["n_obs", "n_nans"]].plot(kind="bar", ax=ax2, alpha=0.4)
				ax.plot(
					_psi_tab[self.psi_str].index,
					_psi_tab[self.psi_str],
					marker="o",
					label=self.psi_str,
					color="black",
					lw=1.5,
					alpha=0.8,
				)

				ax.tick_params(axis="x", labelrotation=90)

				ax2.legend(loc="upper left")
				ax.legend(loc="upper right")
				plt.title(_name)
				plt.tight_layout()

				if to_save:
					pdf.savefig()
					plt.close()
				else:
					plt.show()

	def calculation_to_excel(
		self,
		psi_res,
		psi_bin_counts=None,
		filepath="psi_calc.xlsx",
		drop_vars=None,
		bin_symbols=3,
		drop_bins=None,
		stats_to_save=None
	):
		"""Save calculation result to excel
		Parameters
		-----------
		psi_res: dict
			Result of self.calculte method
		filepath: str or Path
			Path with filename to save excel file.
		drop_vars: list
			Variables to delete from report
		bin_symbols: int
			Number of symbols in bin representation text (rounding).
		stats_to_save: list, default None = ['psi']
			Which stats from psi_res to save
		"""
		drop_vars = [] if drop_vars is None else drop_vars
		drop_bins = [] if drop_bins is None else drop_bins
		stats_to_save = [self.psi_str] if stats_to_save is None else stats_to_save
		result_dfs = {}
		for _stat in stats_to_save:
			vars_res = self.make_wide_table(psi_res, use_stat=_stat)
			vars_res = vars_res.loc[~vars_res.index.isin(drop_vars)]  # drop vars
			result_dfs[_stat] = vars_res

		if psi_bin_counts is not None:
			bin_res = self.beautify_bin_distr_table(
				psi_bin_counts, drop_bins=drop_bins, bin_symbols=bin_symbols
			)
			bin_res = bin_res.loc[
				~bin_res.index.get_level_values(0).isin(drop_vars)
			]  # drop vars
			result_dfs["psi_bin_counts"] = bin_res

		if filepath is not None:
			with pd.ExcelWriter(filepath, mode="w") as writer:
				for sheet_name, stat in result_dfs.items():
					stat.to_excel(writer, sheet_name=sheet_name,)
		return result_dfs

	################# BINNING ##################################

	def bin_variable(self, variable, n_bins=None, bins=None):
		"""Check if numerical or categorical and call binning function"""
		variable = pd.Series(variable)
		nan_mask = variable.isna()

		# numerical
		if pd.api.types.is_numeric_dtype(variable):
			binned_var, bins = self._bin_numerical_var(
				variable, n_bins=n_bins, bins=bins
			)

		# categorical
		else:
			binned_var, bins = self._bin_categorical_var(
				variable, n_bins=n_bins, bins=bins,
			)

		# adding missing and out of interval values
		binned_var = binned_var.cat.add_categories([self.na_str, self.out_int_str])
		binned_var[nan_mask] = self.na_str
		binned_var[binned_var.isna()] = self.out_int_str

		return binned_var, bins

	def _bin_numerical_var(self, variable: pd.Series, n_bins, bins):
		"""Performs different binning depending on fit/predict stage or method"""
		# FIT - finding bins
		include_lowest = False
		# print(variable.name)
		if bins is None:
			bins = self._fixed_qcut(variable, n_bins=n_bins, bins=None)
			# adding std to bin edges (min/max values)
			min_bin_edge_std = max([self.min_left_std, self.fit_data[self._vn]["bin_edge_std"][0]])
			bins_std = np.std(bins[~np.isinf(bins)])
			if not np.isinf(bins[0]):
				# + self.left_minv - return to original bin value
				bins[0] = bins[0] + self.left_minv - min_bin_edge_std * bins_std
			else:
				include_lowest = True
			if bins[0] == bins[1]: # appears in low unique and in some corner cases
				bins[0] = bins[0] - self.left_minv

			if not np.isinf(bins[-1]):
				bins[-1] = bins[-1] + self.fit_data[self._vn]["bin_edge_std"][1] * bins_std

			self.fit_data[self._vn]["n_bins"] = len(bins) - 1

			if n_bins != self.fit_data[self._vn]["n_bins"]:
				logging.info(
					f"[INFO] Changed n_bins for {self._vn} from {n_bins} to {self.fit_data[self._vn]['n_bins']}"
				)
		binned_var = pd.cut(variable, bins=bins, include_lowest=include_lowest)

		return binned_var, bins

	def _fixed_qcut(self, variable, n_bins, bins):
		"""Working with few unique values and uneven distributions"""
		if self.fit_data[self._vn]["low_unique"]:
			pass

		# low unique values
		elif variable.nunique() <= n_bins:
			self.fit_data[self._vn]["low_unique"] = True
			# make bins by hand
			bins = variable.unique()
			if len(bins) == 1:
				logger.warning(
					f"Variable {variable.name} is constant={bins[0]} on period. SI failed"
				)
			bins = np.sort(bins[~pd.isnull(bins)]).astype(float)
			# append min value
			bins = np.append(bins[0] - self.left_minv, bins)

		# qcut
		else:
			bins = self.adaptive_qcut(variable, n_bins)

		return bins

	def _bin_categorical_var(self, variable: pd.Series, n_bins, bins):
		"""Performs different binning depending on fit/predict stage"""

		binned_var = variable.copy()

		# FIT
		if bins is None:
			counts = binned_var.value_counts(normalize=True).sort_values()
			counts = counts[counts >= self.min_bin_size_cat]
			binned_var.loc[~binned_var.isin(counts.index)] = np.nan

			if len(counts) > n_bins:
				# print(f"[INFO] Found too many categories if categorical variable, decreasing")
				most_freq_cats = counts.iloc[:n_bins].index
				binned_var.loc[~binned_var.isin(most_freq_cats)] = np.nan

			bins = binned_var.unique()

		# PREDICT OR USER BINS
		else:
			# remove unknown categories
			binned_var.loc[~binned_var.isin(bins)] = np.nan
		bins = np.array(bins)

		raw_cat = pd.Categorical(
			binned_var, categories=bins[~pd.isnull(bins)], ordered=False
		)
		binned_var = pd.Series(raw_cat, index=variable.index, name=variable.name)

		return binned_var, bins

	############### PSI calculations ##########################

	def _normalised_counts(self, var, targ_var=None):
		base_counts = var.value_counts(normalize=False, sort=False, dropna=False).sort_index()
		base_counts.name = var.name
		# If not PSI but some target mean stability
		if targ_var is not None:
			# normalisation by expected in PSI calc
			aggs = targ_var.groupby(var, observed=False).mean().sort_index()
			aggs.name = var.name
		else:
			aggs = base_counts.copy()
			aggs = aggs / aggs.sum()
		return aggs, base_counts

	def _make_psi_table(self, var_cnt_exp, var_cnt_obs, target_mode, base_counts):
		psi_tab = var_cnt_exp.to_frame()
		psi_tab.columns = ["var_exp"]
		psi_tab["var_obs"] = var_cnt_obs.values

		# fixing division by 0
		# set half of observation to all groups without obseravations
		zero_mask = psi_tab == 0
		psi_tab[zero_mask] = 0.5 / self.fit_data[self._vn]["expected_len"]

		# not PSI mode. Normalisation by expected variable
		if target_mode:
			var_exp = psi_tab["var_exp"] / psi_tab["var_exp"].sum()
			var_obs = psi_tab["var_obs"] / psi_tab["var_exp"].sum()

			psi_tab[self.psi_str] = (var_obs - var_exp) * np.log(var_obs / var_exp)
			# weight by segment size

			weight = (base_counts / base_counts.sum()) * (psi_tab[self.psi_str] > 0).sum()
			psi_tab[self.psi_str] = psi_tab[self.psi_str] * weight
		else:  # standard psi
			psi_tab[self.psi_str] = (psi_tab["var_obs"] - psi_tab["var_exp"]) * np.log(
				psi_tab["var_obs"] / psi_tab["var_exp"]
			)

		# returning to 0
		psi_tab[zero_mask] = 0

		psi_tab["bins_obs"] = var_cnt_obs.index

		return psi_tab

	def _calculate_total_psi(self, psi_tab, exclude_miss=False, exclude_out_int=False):
		calc_tab = psi_tab.copy()

		if exclude_miss:
			calc_tab.loc[self.na_str, self.psi_str] = 0
		if exclude_out_int:
			calc_tab.loc[self.out_int_str, self.psi_str] = 0

		return calc_tab[self.psi_str].sum()

	#################### Adaptive qcut ######################################

	def adaptive_qcut(self, variable, q):
		"""variable n_unique should be higher than q"""
		min_bucket_size = (len(variable) / q) * self.min_bin_coeff

		bins, bucket_sizes = find_adaptive_qcut_bins(variable, q, self.left_minv)

		while (bucket_sizes.min() < min_bucket_size) & (q > 1):
			q = q - 1
			bins, bucket_sizes = find_adaptive_qcut_bins(variable, q)

		return bins

	############# other utils ##################

	def _update_calc_counts_tab(self, var_name, new_counts, base_counts, fit=False):
		"""updating calc_bins_counts"""
		if fit:
			# initiating counts variable in dataframe
			self.calc_bins_counts[var_name] = pd.DataFrame(index=new_counts.index)
			self.calc_bins_counts[var_name]["fit"] = new_counts.values
		elif (self.calc_bins_counts.get(var_name) is None):
			self.calc_bins_counts[var_name] = pd.DataFrame(index=new_counts.index)
			self.calc_bins_counts[var_name][new_counts.name] = new_counts.values
		else:
			self.calc_bins_counts[var_name][new_counts.name] = new_counts.values

		if fit:
			self.calc_bins_nobs[var_name] = pd.DataFrame(index=base_counts.index)
			self.calc_bins_nobs[var_name]["fit"] = base_counts.values
		elif (self.calc_bins_nobs.get(var_name) is None):
			self.calc_bins_nobs[var_name] = pd.DataFrame(index=base_counts.index)
			self.calc_bins_nobs[var_name][base_counts.name] = base_counts.values
		else:
			self.calc_bins_nobs[var_name][base_counts.name] = base_counts.values

	########### Output formatting utils ##################

	def _beautify_index_bins(self, index, symbols=3):
		"""takes array with pd.Intervals and formats them to strings
		with specified N symbols in any number"""
		new_index = []
		for _val in index:
			try:
				if isinstance(_val, pd.Interval):
					_new = self.str_truncate_round(_val.left, symbols)
					_new = _new + ": " + self.str_truncate_round(_val.right, symbols)
				else:
					_new = str(_val)
			except:
				_new = _val
			new_index.append(_new)

		return new_index

	@staticmethod
	def str_truncate_round(val, symbols):
		"""Utility for rounding.
		Tryes to keep specified N symbols in number.
		Minus sign and 1e+N notations can go over N symbols
		"""
		if isinf(val):
			return str(val)

		if abs(val) >= 1:
			nd = len(f"{abs(val):.0f}")
		else:
			nd = 1

		if nd >= (symbols + 2):
			return f"{round(val / 10**(nd-1), 1)}e+{nd}"
		elif symbols <= nd < (symbols + 2):
			val = int(round(val))
		else:
			val = round(val, max(symbols - nd, 0))

		return str(val)


	# @staticmethod
	def filter_psi_conditions(self, psi_res, single_thresh, n_bad_thresh=(3, 0.3), mean_bad_thresh=(5, 0.2)):
		"""Utility to filter columns by StabilityIndexCalculator results
		Parameters
		--------------
		psi_res: StabilityIndexCalculator result
		single_thresh: float.
			Single period filter threshold
		n_bad_thresh: tuple. (int, float)
			N highest psi periods to check. If all higher then thresh -> filter.
		mean_bad_thresh: tuple. (int, float)
			If mean PSI higher then thresh in N highest PSI periods -> filter.
		"""
		filter_cols = []

		for name, psi_tab in psi_res.items():
			psi = psi_tab['psi'].sort_values(ascending=False)

			if psi.iloc[0] > single_thresh:
				filter_cols.append(name)
			elif (psi.iloc[0: n_bad_thresh[0]] > n_bad_thresh[1]).all():
				filter_cols.append(name)
			elif psi.iloc[0: mean_bad_thresh[0]].mean() > mean_bad_thresh[1]:
				filter_cols.append(name)
			else:
				pass

		return filter_cols


# ADAPTIVE QCUT
def find_adaptive_qcut_bins(variable, q, left_minv=0.0001):
	"""
	Algorithm for finding optimal quantile bins.
	Made for cases when one value has more than q repetions in variable.
	This creates 'duplicate bins' for pandas.qcut and decreases q.

	Algorithm first assignes bins to quantiles as pd.qcut, but then instead of dropping
	duplicate bins moves them to better positions if possible
	"""
	var = np.array(variable)
	# not working with nans, they will stay inplace
	var_notna = var[~np.isnan(var)].copy()
	var_sorted = np.sort(var_notna)

	# get counts
	var_unique, var_counts = np.unique(var_sorted, return_counts=1)
	var_counts = np.append([0], var_counts)
	var_unique = np.append([var_unique[0] - left_minv], var_unique)

	assert (var_unique == np.sort(var_unique)).all()

	var_cnt_csum = var_counts.cumsum()
	expected_len = len(var_sorted) / q

	# initiate bins (qcut on unique values)
	bins_cs_idxs = initiate_bins(q, var_cnt_csum, var_unique, expected_len)

	# adaptive correction of initiated bins
	bins_cs_idxs = qcut_correction(q, bins_cs_idxs, var_cnt_csum)

	bins = var_unique[bins_cs_idxs]
	bucket_sizes = var_cnt_csum[bins_cs_idxs][1:] - var_cnt_csum[bins_cs_idxs][:-1]

	bins = np.sort(np.unique(bins))
	return bins, bucket_sizes

def initiate_bins(q, var_cnt_csum, var_unique, expected_len):
	norm_cnt_csum = var_cnt_csum / expected_len
	# look for closest value to expected_len
	bins_cs_idxs = [0]
	for i in range(1, q):
		_idx = (norm_cnt_csum < i).sum()
		if abs(norm_cnt_csum[_idx - 1] - i) > abs(norm_cnt_csum[_idx] - i):
				bins_cs_idxs.append(_idx)
		else:
			bins_cs_idxs.append(_idx - 1)
	bins_cs_idxs.append(len(var_unique) - 1)
	# or_bins = np.sort(np.unique(var_unique[bins_cs_idxs]))

	# if qcut failed with duplicate edges, use unique elements
	if len(bins_cs_idxs) != len(np.unique(bins_cs_idxs)):
		added_bins = np.unique(bins_cs_idxs).copy()
		n_add = len(bins_cs_idxs) - len(np.unique(bins_cs_idxs)) # how many bins to add
		for n in range(n_add):
			bucket_sizes = var_cnt_csum[added_bins][1:] - var_cnt_csum[added_bins][:-1] # difference between counts
			for _idx_insert in np.argsort(bucket_sizes)[::-1] + 1:
				if added_bins[_idx_insert] - added_bins[_idx_insert - 1] > 1: # if has unique values between edges
					if added_bins[_idx_insert] - added_bins[_idx_insert - 1] == 2:
						_val_insert = int(np.round(np.mean([added_bins[_idx_insert], added_bins[_idx_insert - 1]])))
					else:
						_il, _ir = added_bins[_idx_insert - 1], added_bins[_idx_insert]
						_arr_counts = var_cnt_csum[_il + 1: _ir]
						_elen = var_cnt_csum[_il] + (var_cnt_csum[_ir] - var_cnt_csum[_il]) / 2
						# choose left or right value
						_idx = int(np.clip((_arr_counts < _elen).sum(), 0, len(_arr_counts) - 1))
						if abs(_arr_counts[_idx - 1] - _elen) > abs(_arr_counts[_idx] - _elen):
							_val_insert = _il + _idx
						else:
							_val_insert = _il + _idx + 1
					break
			# print(_idx_insert, _val_insert, var_unique[_val_insert])
			added_bins = np.insert(added_bins, _idx_insert, _val_insert)

		bins_cs_idxs = np.array(added_bins)
	else:
		bins_cs_idxs = np.array(bins_cs_idxs)

	return bins_cs_idxs

def qcut_correction(q, bins_cs_idxs, var_cnt_csum):
	bins_ids = np.arange(q + 1)
	forbid_move_bins = np.array([])
	metr_bucket_sizes = []
	counter = 0
	# while all bins expect edges are not forbidden to move
	while (counter < q * 10) & (len(forbid_move_bins) != len(bins_ids) - 2):
		# get sizes of buckets
		bucket_sizes = (
			var_cnt_csum[bins_cs_idxs][1:] - var_cnt_csum[bins_cs_idxs][:-1]
		)
		bucket_sizes_diff = np.abs(bucket_sizes[:-1] - bucket_sizes[1:])

		metr_bucket_sizes.append(np.std(bucket_sizes))

		# reindex bin ids for bucket_sizes
		forbid_buckets_idxs = np.unique(
			np.clip(forbid_move_bins - 1, 0, max(bins_ids) - 1)
		).astype(int)
		# set forbidden bins to 0
		if len(forbid_buckets_idxs) > 0:
			bucket_sizes_diff[forbid_buckets_idxs] = 0

		# if nothing to move
		if bucket_sizes_diff.max() == 0:
			break

		# worse position
		move_bin_id = bins_ids[bucket_sizes_diff.argmax() + 1]

		prev_pos_idx = bins_cs_idxs[move_bin_id]

		# find middle of the segment and closest array value to it
		middle_cs_val = (
			var_cnt_csum[
				[bins_cs_idxs[move_bin_id - 1], bins_cs_idxs[move_bin_id + 1]]
			].sum()
			/ 2
		)
		new_pos_idx = np.abs(var_cnt_csum - middle_cs_val).argmin()

		# if moved then releasing neighboors from forbid_move_bins
		if prev_pos_idx != new_pos_idx:
			bins_cs_idxs[move_bin_id] = new_pos_idx
			forbid_move_bins = forbid_move_bins[
				~np.isin(forbid_move_bins, [move_bin_id - 1, move_bin_id + 1])
			]

		# always add last moded/attempted bin to forbidden
		forbid_move_bins = np.unique(np.append(forbid_move_bins, move_bin_id))

		counter += 1

	return bins_cs_idxs



def psi_plot(psi_res: dict, n_cols=5, figsize=(24, 4), save_path=None):

    psi_str = "psi"
    hlines=(0.1, 0.2)
    y_lim=(-0.05, 0.5)

    # Определяем структуру таблицы графиков
    n_plots = len(psi_res.items())
    n_cols = 5  # Количество столбцов
    n_rows = (n_plots + n_cols - 1) // n_cols  # Вычисляем необходимое количество строк

    # Создаем таблицу графиков
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    fig.suptitle('PSI Analysis', fontsize=16)

    # Если только одна строка, axes будет 1D массивом
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for idx, (_name, psi_tab) in enumerate(psi_res.items()):
        if idx >= len(axes):  # На случай, если графиков больше, чем ячеек
            break

        ax = axes[idx]
        ax2 = ax.twinx()

        _psi_tab = psi_tab.copy()
        _psi_tab.index = _psi_tab.index.astype(str)

        ax.set_ylim(y_lim)
        ax.axhline(hlines[0], ls="-.", c="g")
        ax.axhline(hlines[1], ls="-.", c="r")
        ax.axhline(0, ls="-.", c="blue", lw=1, alpha=0.5)

        _psi_tab[["n_obs", "n_nans"]].plot(kind="bar", ax=ax2, alpha=0.4)
        ax.plot(
            _psi_tab[psi_str].index,
            _psi_tab[psi_str],
            marker="o",
            label=psi_str,
            color="black",
            lw=1.5,
            alpha=0.8,
        )

        ax.tick_params(axis="x", labelrotation=90)
        ax2.legend(loc="upper left")
        ax.legend(loc="upper right")
        ax.set_title(_name)

    # Скрываем пустые subplots, если они есть
    for idx in range(len(psi_res.items()), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    plt.show()
