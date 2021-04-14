# --- Do not remove these libs ----------------------------------------------------------------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import logging
from pandas import DataFrame, DatetimeIndex, merge
from freqtrade.strategy import IStrategy
from typing import Dict, List
from skopt.space import Dimension, Real
# ---------------------------------------------------------------------------------------------------------------------

# --- logger for parameter merging output, only remove if you remove it further down too! -----------------------------
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------------------------------------------------


class QuickieHyperStrategy(IStrategy):
    """
    enhanced auto hyperoptable version based on
    https://github.com/back8/github_freqtrade_freqtrade-strategies/blob/master/user_data/strategies/berlinguyinca/ReinforcedQuickie.py

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! as of today (14.04.2021) you need the freqtrade/develop version to be able     !!!
    !!! to run hyperopt/backtest with this new strategy format                         !!!
    !!!                                                                                !!!
    !!! please check https://github.com/freqtrade/freqtrade/pull/4596 for further      !!!
    !!! information about the new auto-hyperoptable strategies!                        !!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    original author@: Gert Wohlgemuth

    idea:
        momentum based strategie. The main idea is that it closes trades very quickly, while avoiding excessive losses. Hence a rather moderate stop loss in this case
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "100": 0.01,
        "30": 0.03,
        "15": 0.06,
        "10": 0.15,
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.25

    # Optimal timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['adx'] = ta.ADX(dataframe)

        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > 30) &
                    (dataframe['tema'] < dataframe['bb_middleband']) &
                    (dataframe['tema'] > dataframe['tema'].shift(1)) &
                    (dataframe['sma_200'] > dataframe['close'])

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > 70) &
                    (dataframe['tema'] > dataframe['bb_middleband']) &
                    (dataframe['tema'] < dataframe['tema'].shift(1))
            ),
            'sell'] = 1
        return dataframe

    # nested hyperopt class    
    class HyperOpt:
        # defining buy / sell indicator spaces as dummy, 
        # so that no error is thrown about missing sell indicators
        # when hyperopting for all spaces
        @staticmethod
        def indicator_space() -> List[Dimension]:
            return []

        @staticmethod
        def sell_indicator_space() -> List[Dimension]:
            return []
        
        # custom stop loss range
        @staticmethod
        def stoploss_space() -> List[Dimension]:
            return [
                Real(-0.25, -0.01, name='stoploss'),
            ]
