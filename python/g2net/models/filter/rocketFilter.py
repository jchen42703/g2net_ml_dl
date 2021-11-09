from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from g2net.io.dataset import G2NetDataset
class Filter():
    def __init__():
        super().__init__()

def load_train_csv(dir_to_csv):
    return pd.DataFrame(pd.read_csv(dir_to_csv))

""""Method to split training data
*arg df, data frame from load_train_csv

returns train, val , test dataframes
"""
def split_data(df):
    divideLength1 = round(len(df) * .9)
    divideLength2 = round(len(df)* .95)
    train = df[:divideLength1]
    val = df[divideLength1:divideLength2]
    test = df[divideLength2:]
    return train,val ,test
"""Method to train model 
*arg train, training df from split_data

returns classifier used to predict 
"""
def train(train):
    X_train = train['id']
    signals = []
    targets = []
    for i in X_train:
        signal,target = G2NetDataset.__getitem__(i)
        signals.append(signal)
        targets.append(target)
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
def predict(classifier, data):
    ids = data['id']
    signals = []
    for i in ids:
        signal,target = G2NetDataset.__getitem__(i)
        signals.append(signal)
    parameters = fit(signals)
    X_data = transform(signals,parameters)
    return classifier.predict(X_data)

