# --- Do not remove these libs ----------------------------------------------------------------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import logging
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter, RealParameter
from typing import Dict, List
from skopt.space import Dimension
# ---------------------------------------------------------------------------------------------------------------------

# --- logger for parameter merging output, only remove if you remove it further down too! -----------------------------
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------------------------------------------------

class CombinedBinHAndClucHyperStrategy(IStrategy):

    """
    enhanced auto hyperoptable version based on
    https://github.com/freqtrade/freqtrade-strategies/blob/master/user_data/strategies/berlinguyinca/CombinedBinHAndCluc.py

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! as of today (14.04.2021) you need the freqtrade/develop version to be able     !!!
    !!! to run hyperopt/backtest with this new strategy format                         !!!
    !!!                                                                                !!!
    !!! please check https://github.com/freqtrade/freqtrade/pull/4596 for further      !!!
    !!! information about the new auto-hyperoptable strategies!                        !!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Based on a backtesting:
    - the best perfomance is reached with "max_open_trades" = 2 (in average for any market),
      so it is better to increase "stake_amount" value rather then "max_open_trades" to get more profit
    
    - if the market is constantly green(like in JAN 2018) the best performance is reached with
      "max_open_trades" = 2 and minimal_roi = 0.01
    """

    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.05

    timeframe = '5m'

    # derived from EMA slow
    startup_candle_count = 50

    # default by parameters
    buy_params = {
        'buy_bin_bbdelta_close': 0.008,
        'buy_bin_closedelta_close': 0.0175,
        'buy_bin_tail_bbdelta': 0.25,
        'buy_cluc_close_bblowerband': 0.985,
        'buy_cluc_volume': 20
    }

    sell_params = { } # not used but defined for security reasons

    # for hyperopt
    buy_bin_bbdelta_close =  RealParameter(0.0, 0.02, default=0.008, space='buy', optimize=True, load=True)
    buy_bin_closedelta_close = RealParameter(0.0, 0.03, default=0.0175, space='buy', optimize=True, load=True)
    buy_bin_tail_bbdelta = RealParameter(0.0, 1.0, default=0.25, space='buy', optimize=True, load=True)
    buy_cluc_close_bblowerband = RealParameter(0.0, 1.5, default=1.5, space='buy', optimize=True, load=True)
    buy_cluc_volume = IntParameter(10, 40, default=20, space='buy', optimize=True, load=True)

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        try:
            from mergedeep import merge
        except ImportError as error:
            # Output expected ImportErrors.
            logger.info("could not import mergedeep, please check if pip is installed: %s", error)
            logger.info("therefor we are not able to merge parameters from config")
        else:
            logger.info('mergedeep found, so attempting to find strategy parameters in config file')
            if self.config.get('strategy_parameters', {}).get(self.__class__.__name__, False):
                cfg_strategy_parameters = self.config.get('strategy_parameters', {}).get(self.__class__.__name__, False)
                logger.info('strategy_parameters from config: %s', repr(cfg_strategy_parameters))
                if cfg_strategy_parameters.get('buy_params', {}):
                    logger.info('merging buy_params from config: %s', cfg_strategy_parameters.get('buy_params'))
                    merge(self.buy_params, cfg_strategy_parameters.get('buy_params'))
                if cfg_strategy_parameters.get('sell_params', {}):
                    logger.info('merging sell_params from config: %s', cfg_strategy_parameters.get('sell_params'))
                    merge(self.sell_params, cfg_strategy_parameters.get('sell_params'))
            else:
                logger.info('no strategy_parameters found in config')
            logger.info('final buy_params: %s', repr(self.buy_params))
            logger.info('final sell_params: %s', repr(self.sell_params))
    
    @staticmethod
    def bollinger_bands(stock_price, window_size, num_of_std):
        rolling_mean = stock_price.rolling(window=window_size).mean()
        rolling_std = stock_price.rolling(window=window_size).std()
        lower_band = rolling_mean - (rolling_std * num_of_std)
        return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # strategy BinHV45
        mid, lower = self.bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        # strategy ClucMay72018
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (  # strategy BinHV45
                    dataframe['lower'].shift().gt(0) &
                    dataframe['bbdelta'].gt(dataframe['close'] * self.buy_bin_closedelta_close.value) &
                    dataframe['closedelta'].gt(dataframe['close'] * self.buy_bin_closedelta_close.value) &
                    dataframe['tail'].lt(dataframe['bbdelta'] * self.buy_bin_tail_bbdelta.value) &
                    dataframe['close'].lt(dataframe['lower'].shift()) &
                    dataframe['close'].le(dataframe['close'].shift())
            ) |
            (  # strategy ClucMay72018
                    (dataframe['close'] < dataframe['ema_slow']) &
                    (dataframe['close'] < self.buy_cluc_close_bblowerband.value * dataframe['bb_lowerband']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * self.buy_cluc_volume.value))
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_middleband']),
            'sell'
        ] = 1
        return dataframe

    # nested hyperopt class    
    class HyperOpt:

        # defining as dummy, so that no error is thrown about missing
        # sell indicator space when hyperopting for all spaces
        @staticmethod
        def sell_indicator_space() -> List[Dimension]:
            return []