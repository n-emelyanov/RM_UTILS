
import itertools
import pandas as pd
import numpy as np
from typing import List
from IPython.display import display



def all_combinations(my_list):
    return itertools.chain.from_iterable(
        itertools.combinations(my_list, i + 1) for i in range(len(my_list)))


class MetricCalculator:
    """Source для расчета метрик по группам.

    Parameters:
    ---
    metr_funcs : dict
        Словарь функций, используемых для расчета метрик по группам.
        Функции должны следовать конвенции sklearn - первые 2 аргумента = y_true, y_pred

    stats_funcs : dict, default={}
        Словарь функций для расчета метрик по группам.
        В отличие от metr_funcs используется для расчета статистик по таргету или по другим столбцам.
        Функции принимают два аргумента - (y_true, data=None)

    """

    def __init__(
        self,
        metr_funcs: dict,
        funcs_params: dict = {},
        stats_funcs: dict = {}
    ):
        self.metr_funcs = metr_funcs
        self.funcs_params = funcs_params
        self.stats_funcs = stats_funcs


    def _partial_stack(
        self, result: pd.DataFrame, pred_cols: List[str], group_cols: List[str]
    ) -> pd.DataFrame:
        """Преобразует DataFrame с MultiIndex колонками в частично 'stacked' формат."""

        columns_to_stack = pd.MultiIndex.from_product(
            [pred_cols, self.metr_funcs.keys()]
        )

        # Стакаем и меняем столбцы
        stacked = (
            result[columns_to_stack]
            .stack(level=0)
            .reindex(self.metr_funcs.keys(), axis=1)
            .reset_index()
            .rename(columns={f"level_{len(group_cols)}": "pred"})
            .set_index(group_cols)
        )

        kept = result.drop(columns=columns_to_stack)
        kept.columns = kept.columns.get_level_values(1)

        if len(kept.columns) == 0:
            return stacked.reset_index()

        # return pd.concat([stacked, kept], axis=1).reset_index()
        return stacked.join(kept).reset_index()



    def _set_metr_funcs(self, data: pd.DataFrame, pred_cols: List[str]) -> dict:

        # Используем pd.Series таргета для ускорения
        true_values = data[self.true_col]
        agg_funcs = {}

        # Оборачиваем metr_funcs, подставляем параметры если имеются, добавляем в agg_funcs
        for pred_col in pred_cols:
            agg_funcs[pred_col] = [
                (
                    func_name,
                    lambda x, f=func, params=self.funcs_params.get(func_name, {}): f(
                        true_values[x.index], x, **params
                    ),
                )
                for func_name, func in self.metr_funcs.items()
            ]

        # Оборачиваем stats_funcs и добавляем в agg_funcs
        if self.stats_funcs:
            agg_funcs[self.true_col] = [
                (
                    func_name,
                    lambda x, func=func: func(true_values[x.index], data.loc[x.index])
                )
                    for func_name, func in self.stats_funcs.items()
            ]

        return agg_funcs

    def calculate(
        self,
        data: pd.DataFrame,
        true_col: str,
        pred_cols: List[str] | str,
        group_cols: List[str] | str,
        groupby_exclude_combinations: list[str] | None = None,
        pretify_one_func: bool = False,
    ) -> pd.DataFrame:
        """Расчет метрик по группам.

        Parameters
        ---
        data : pd.DataFrame
            Датафрейм с данными.
        true_col : str
            Имя столбца с истинными значениями (передается первым аргументом в metr_funcs).
        pred_cols : List[str] | str
            Список имен столбцов с предсказаниями или одно имя столбца.
        group_cols : List[str]
            Список имен столбцов для группировки.
        groupby_exclude_combinations : list[str] or None, default=None
            Список столбцов для дополнитльной агрегации с all.
        pretify_one_func : bool, default=False
            Если True, то возвращает DataFrame с одной функцией в колонке.

        Returns
        -------
        pd.DataFrame
            Датафрейм с рассчитанными метриками.

        Examples
        -------
        >>> metr_calc = MetricCalculator(
        ...     metr_funcs={'mae': mean_absolute_error},
        ...     stats_funcs={'n_obs': lambda y_true, data: len(data)
        ... )
        >>> res = metr_calc.calculate(
        ...     data=data,
        ...     true_col='target',
        ...     pred_cols=['pred1', 'pred2'],
        ...     group_cols=['group1', 'group2']
        ... )
        """
        self.true_col = true_col

        # Конвертируем pred_cols/group_cols в список, если это строка
        pred_cols = [pred_cols] if isinstance(pred_cols, str) else pred_cols
        group_cols = [group_cols] if isinstance(group_cols, str) else group_cols

        agg_funcs = self._set_metr_funcs(data, pred_cols)

        result = data.groupby(group_cols).agg(agg_funcs)

        if groupby_exclude_combinations is not None:

            for idxs in all_combinations(range(len(groupby_exclude_combinations))):

                # Исключенные группы
                not_groupby_cols = list(
                    np.array(groupby_exclude_combinations.copy())[list(idxs)]
                )
                groupby_cols = [
                    col for col in group_cols if col not in not_groupby_cols
                ]

                result_temp = data.groupby(groupby_cols).agg(agg_funcs)

                for _removed in not_groupby_cols:
                    result_temp[_removed] = 'all'

                result_temp = result_temp.reset_index().set_index(group_cols)
                result = pd.concat([result, result_temp], axis=0)

        # Красивый вывод, если была задействована одна функция
        if pretify_one_func and len(self.metr_funcs) == 1:

            result.columns = pred_cols + [*(self.stats_funcs or {})]
            return result.reset_index()

        result = self._partial_stack(result, pred_cols, group_cols)

        return result
