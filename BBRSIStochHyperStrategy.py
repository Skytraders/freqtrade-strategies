# --- Do not remove these libs ----------------------------------------------------------------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import logging
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter, CategoricalParameter
from typing import Dict, List
from skopt.space import Dimension, Real
# ---------------------------------------------------------------------------------------------------------------------

# --- logger for parameter merging output, only remove if you remove it further down too! -----------------------------
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------------------------------------------------

class BBRSIStochHyperStrategy(IStrategy):

    """
    enhanced auto hyperoptable version based on
    https://github.com/faGH/fa.services.plutus/blob/main/user_data/strategies/fa_m31h_strategy.py

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! as of today (14.04.2021) you need the freqtrade/develop version to be able     !!!
    !!! to run hyperopt/backtest with this new strategy format                         !!!
    !!!                                                                                !!!
    !!! please check https://github.com/freqtrade/freqtrade/pull/4596 for further      !!!
    !!! information about the new auto-hyperoptable strategies!                        !!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    This is FrostAura's mark 3 strategy which aims to make purchase decisions
    based on the BB, RSI and Stochastic.

    """

    # user hyperopt for best minimal roi and stoploss!
    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.05

    # best timeframes currently in backtest are 15m and 1h
    timeframe = '15m'

    startup_candle_count = 50

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # default buy parameters
    buy_params = {
        'buy_stoch_enabled': True,
        'buy_stoch_value': 25,
        'buy_rsi_value': 30,
        'buy_bb_trigger': 'bb_lowerband1'
    }

    # default sell parameters
    sell_params = { 
        'sell_rsi_value': 30,
        'sell_bb_trigger': 'bb_middleband1'
        
    } 

    # --- hyperopt parameters > can be selectively turned on/off via optimize=True/False ------------------------------
    buy_stoch_enabled =  CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_stoch_value = IntParameter(5, 30, default=25, space='buy', optimize=True, load=True)
    buy_rsi_value = IntParameter(5, 30, default=30, space='buy', optimize=True, load=True)
    buy_bb_trigger = CategoricalParameter(
        [
            'bb_lowerband1', 
            'bb_lowerband2', 
            'bb_lowerband3', 
            'bb_lowerband4'
        ], 
        default='bb_lowerband1', space='buy', optimize=True, load=True)
    
    sell_rsi_value = IntParameter(40, 90, default=75, space='sell', optimize=True, load=True)
    sell_bb_trigger = CategoricalParameter(
        [
            'bb_lowerband1', 
            'bb_middleband1', 
            'bb_upperband1'
        ], 
        default='bb_upperband1', space='sell', optimize=True, load=True)

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
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        
        # Stochastic Slow
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        # Bollinger bands
        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband1'] = bollinger1['lower']
        dataframe['bb_middleband1'] = bollinger1['mid']
        dataframe['bb_upperband1'] = bollinger1['upper']
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']
        bollinger4 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband4'] = bollinger4['lower']
        dataframe['bb_middleband4'] = bollinger4['mid']
        dataframe['bb_upperband4'] = bollinger4['upper']
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (        
                    ( # stoch enabled
                        (self.buy_stoch_enabled.value == True) &
                        (dataframe['slowd'] > self.buy_stoch_value.value) &
                        (dataframe['slowk'] > self.buy_stoch_value.value)
                    ) | # stoch disabled
                    (self.buy_stoch_enabled.value == False) 
                ) &
                ( 
                    (dataframe['rsi'] > self.buy_rsi_value.value) &
                    (dataframe['slowk'] < dataframe['slowd']) &
                    (dataframe["close"] < dataframe[self.buy_bb_trigger.value])
                )
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['slowk'] < dataframe['slowd']) &
                (dataframe['rsi'] > self.sell_rsi_value.value) &
                (dataframe["close"] > dataframe[self.sell_bb_trigger.value])
            ),
            'sell'
        ] = 1
        return dataframe

    # nested hyperopt class    
    class HyperOpt:

        # custom stop loss range
        @staticmethod
        def stoploss_space() -> List[Dimension]:
            return [
                Real(-0.5, -0.02, name='stoploss'),
            ]