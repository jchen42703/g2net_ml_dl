import numpy as np
from sklearn.metrics import roc_auc_score


class MetricTemplate:
    '''
    Custom metric template
    # Usage
    general:    Metric()(target, approx)
    catboost:   eval_metric=Metric()
    lightgbm:   metric='Metric_Name', feval=Metric().lgb
    pytorch:    Metric().torch(output, labels)

    Example:
        eval_metric = AUC().torch
        fit_params = {
            'eval_metric': eval_metric,
            ...
        }
        trainer = TorchTrainer(...)
        trainer.fit(**fit_params)
    '''

    def __init__(self, maximize=False):
        self.maximize = maximize

    def __repr__(self):
        return f'{type(self).__name__}(maximize={self.maximize})'

    def _test(self, target, approx):
        # Metric calculation
        pass

    def __call__(self, target, approx):
        return self._test(target, approx)

    ''' CatBoost '''

    def get_final_error(self, error, weight):
        return error / weight

    def is_max_optimal(self):
        return self.maximize

    def evaluate(self, approxes, target, weight=None):
        # approxes - list of list-like objects (one object per approx dimension)
        # target - list-like object
        # weight - list-like object, can be None
        assert len(approxes[0]) == len(target)
        if not isinstance(target, np.ndarray):
            target = np.array(target)

        approx = np.array(approxes[0])
        error_sum = self._test(target, approx)
        weight_sum = 1.0

        return error_sum, weight_sum

    ''' LightGBM '''

    def lgb(self, approx, data):
        target = data.get_label()
        return self.__class__.__name__, self._test(target,
                                                   approx), self.maximize

    lgbm = lgb  # for compatibility
    ''' XGBoost '''

    def xgb(self, approx, dtrain):
        target = dtrain.get_label()
        return self.__class__.__name__, self._test(target, approx)

    ''' PyTorch '''

    def torch(self, approx, target):
        return self._test(target.detach().cpu().numpy(),
                          approx.detach().cpu().numpy())


class AUC(MetricTemplate):
    '''
    Area under ROC curve
    '''

    def __init__(self):
        super().__init__(maximize=True)

    def _test(self, target, approx):
        if len(approx.shape) == 1:
            pass
        elif approx.shape[1] == 1:
            approx = np.squeeze(approx)
        elif approx.shape[1] == 2:
            approx = approx[:, 1]
        else:
            raise ValueError(f'Invalid approx shape: {approx.shape}')
        target = np.round(target)
        return roc_auc_score(target, approx)