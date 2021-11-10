import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import MiniRocket
from g2net.io.dataset import G2NetDataset
class Filter():
    def __init__():
        super().__init__()
    
    """Method to train model 
    *arg train, training df from split_data

    returns classifier used to predict 
    """
    def train(signals,targets):
        parameters = fit(signals)
        X_training_transform = transform(signals, parameters)
        classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        classifier.fit(X_training_transform, targets)
        return classifier

    """Method to predict
    *arg classifier, classifier returned from training 
    *arg data, data to predict on 

    returns prediction of data
    """
    def predict(classifier, signals):
        parameters = fit(signals)
        X_data = transform(signals,parameters)
        return classifier.predict(X_data)

    def score(classifier, signal_test, target_test):
        return classifier.score(signal_test,target_test)
