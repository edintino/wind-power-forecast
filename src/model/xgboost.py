import logging
import xgboost as xgb

logger = logging.getLogger(__name__)

class XGBoostModel:
    """
    Wrapper class around XGBoost regressor for single-step time-series forecasting.

    Args:
        **params: Arbitrary keyword arguments for XGBRegressor hyperparameters.

    Methods:
        fit(X_train, y_train, X_val=None, y_val=None): Trains the model with optional validation.
        predict(X_test): Generates predictions for test data.
    """
    def __init__(self, **params):
        super().__init__()
        self.params = params
        self.model_ = xgb.XGBRegressor(**params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("Fitting XGBoost model...")
        if X_val is not None and y_val is not None:
            self.model_.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model_.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model_.predict(X_test)
