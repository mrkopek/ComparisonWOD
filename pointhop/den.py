import numpy as np
import pickle
import pointhop
# Path to your PCD file
pcd_file = "path_to_your_file.pcd"

# Read the PCD file
fea = pickle.load(open("feat.pkl","rb"))
labels = pickle.load(open("label.pkl","rb"))

pred_valid = pickle.load(open("prevalid.pkl","rb"))
feat_tmp_train=fea['train'] 
feat_tmp_valid=fea['test'] 
train_label=labels['train']
valid_label=labels['test']

idx=0



acc = pointhop.average_acc(valid_label, pred_valid[idx])



def average_acc(label, pred_label):

    classes = np.arange(51)
    acc = np.zeros(len(classes))
    for i in range(len(classes)):
        ind = np.where(label == classes[i])[0]
        pred_test_special = pred_label[ind]
        acc[i] = len(np.where(pred_test_special == classes[i])[0])/float(len(ind))
    return acc


def rf_classifier(feat, y):
    '''
    Train svm based on the provided feature.
    :param feat: [num_samples, feature_dimension]
    :param y: label provided
    :return: classifer
    '''
    clf = ensemble.RandomForestClassifier(n_estimators=128, bootstrap=False,
                                          n_jobs=-1)
    clf.fit(feat, y)
    return clf