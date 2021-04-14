
# freqtrade-strategies
## Description
Existing and self-developed strategies, rewritten to support the new HyperStrategy format from the freqtrade-develop branch.

## Status
This is pure development / break'n'fix stuff. Use absolutely at your  own risk. Not recommended to use with live money!
## How To
**IMPORTANT:** All those are  [Auto-HyperOptable Strategies](https://github.com/freqtrade/freqtrade/pull/4596) that only work with the lastest freqtrade develop branch. No more seperate files for hyperopt & strategy are needed. Please make yourself aware of this new format and ensure you are running those with the latest develop branch!

### HyperOpt
Run hyperopt as outlined in the documentation just ommiting the `--hyperopt` parameter, example:
```properties
freqtrade backtesting--config ./user_data/config.json --hyperopt-loss SortinoHyperOptLossDaily --spaces all --strategy CombinedBinHAndClucHyperStrategy --epochs 1000 --timerange 20210301-20210331
```
### Backtest
Can be ran as usual, by providing the same strategy name as in hyperopt above.
```properties
freqtrade backtesting --strategy CombinedBinHAndClucHyperStrategy --config ./user_data/config.json --timerange 20210101-20210316
```

## Credits
Freqtrade (https://www.freqtrade.io/) is used as the underlying trading framework so all credit to them. This repository aims to provide custom strategies for this framework and create an automated pipeline where the strategies can evolve over nightly builds by ML optimizations running on each build.

## Contribute
In order to contribute, simply fork the repository, make changes and create a pull request.

## Support
join.the.nightshift@protonmail.com