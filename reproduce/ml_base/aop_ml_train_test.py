from ifeature.codes import *
from ifeature.PseKRAAC import *
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, confusion_matrix
from scipy.stats import pearsonr, kendalltau
from sklearn.model_selection import GridSearchCV
import random


xgb_params = {
    "n_estimators": 100,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "log_loss",
    "subsample": 0.8,
}

def isDataFrame(obj):
    if type(obj) == pd.core.frame.DataFrame:
        return True
    else:
        return False
# calculate concordance_correlation_coefficient
def CCC(y_true, y_pred):
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator
def getTrueFastas(fastas):
    if isDataFrame(fastas):
        fts = fastas[["ID", "SEQUENCE"]]
        return fts.to_numpy()
    else:
        return fastas

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'y']
def one_hot_encode(seq, max_len=60):
    # o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))
    l = len(s)
    if max_len < l:
        max_len = l
    x = pd.DataFrame(np.zeros((max_len, 20), dtype=int), columns=codes)
    a = s[0].str.get_dummies(sep=',')
    # a = a.join(x)
    # a = a.sort_index(axis=1)
    b = x + a
    b = b.replace(np.nan, 0)
    b = b.astype(dtype=int)
    # e = a.values.flatten()
    return b

def oneHot(fastas, max_len=60, class_val=None):
    fastas = getTrueFastas(fastas)
    fts = []
    names = fastas[:,0]
    for seq in fastas[:,1]:
        if len(seq) > max_len:
            continue
        # print("seq: ", seq)
        e = one_hot_encode(seq, max_len=max_len)
        e = e.values.flatten()
        fts.append(e)
    df = pd.DataFrame(fts)
    df.index = names
    # df.columns = df.iloc[0]
    print("Gene oneHot:")
    row, col = df.shape
    print("\t\t its class value: %s" % (str(class_val)))
    print("\t\t its input No.: %s" % row)
    print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

#subtype = {'g-gap': 0, 'lambda-correlation': 4}
#subtype = 'g-gap' or 'lambda-correlation'
def genePsekraac(fastas, ft_name="type1", raactype=2, subtype='lambda-correlation', ktuple=2, gap_lambda=1, class_val=None):
    # fastas = readFasta.readFasta(path)
    # gap_lambda = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    fastas = getTrueFastas(fastas)
    #type1(fastas, subtype, raactype, ktuple, glValue)
    eval_func = "%s.type1(fastas, subtype=subtype, raactype=raactype, ktuple=ktuple, glValue=gap_lambda)" % (ft_name)
    print(eval_func)
    encdn = eval(eval_func)
    df = pd.DataFrame(encdn)
    df.index = df.iloc[:, 0]
    df.columns = df.iloc[0]
    df.drop(["#"], axis=1, inplace=True)
    df.drop(["#"], axis=0, inplace=True)
    print("feature number of PseKRAAC.%s(%s, raac_type=%d, ktuple=%d, gap_lambda=%d): %d" %
          (ft_name, subtype, raactype, ktuple, gap_lambda, len(df.columns)))
    ft_whole_name = "%sraac%s" % (ft_name, raactype)
    print("Gene %s :" % (ft_whole_name))
    row, col = df.shape
    print("\t\t its class value: %s" % (str(class_val)))
    print("\t\t its input No.: %s" % row)
    print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

def GeneIfeature(fastas, ft_name="AAC", gap=0, nlag=4, lambdaValue=4, class_val=None):
    # fastas = readFasta.readFasta(path)
    # CKSAAP: gap = 0, 1, 2, 3 (3 = min sequence length - 2)
    # SOCNumber QSOrder PAAC APAAC: lambdaValue = 0, 1, 2, 3, 4 (4 = min sequence length - 1)
    # NMBroto: nlag= 2, 3, 4
    #fastas = getTrueFastas(fastas)
    fastas = readFasta.readFasta(fastas)
    eval_func = "%s.%s(fastas, gap=%d, order=None, nlag=%d, lambdaValue=%d)" % (ft_name, ft_name, gap, nlag, lambdaValue)
    print(eval_func)
    encdn = eval(eval_func)
    df = pd.DataFrame(encdn)
    df.index = df.iloc[:, 0]
    df.columns = df.iloc[0]
    df.drop(["#"], axis=1, inplace=True)
    df.drop(["#"], axis=0, inplace=True)
    # print("%s's feature number: %d" % (ft_name, len(df.columns)))
    print("Gene %s :" % (ft_name))
    row, col = df.shape
    print("\t\t its class value: %s" % (str(class_val)))
    print("\t\t its input No.: %s" % row)
    print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def ml_train_test(x_train, y_train, x_test, y_test, classifier_name):
    # Select classifier based on name
    if classifier_name == "SVM":
        classifier = SVC()
    elif classifier_name == "RF":
        classifier = RandomForestClassifier()
    elif classifier_name == "DT":
        classifier = DecisionTreeClassifier()
    elif classifier_name == "KNN":
        classifier = KNeighborsClassifier()
    elif classifier_name == "XGBoost":
        classifier = GradientBoostingClassifier(**xgb_params)

    # Train the classifier
    classifier.fit(x_train, y_train)

    # Make predictions
    preds = classifier.predict(x_test)

    # Calculate classification metrics
    cls_mcc = matthews_corrcoef(y_test, preds)
    cls_accuracy = accuracy_score(y_test, preds)
    cls_precision = precision_score(y_test, preds)  # Using weighted for multi-class
    cls_sensitivity = recall_score(y_test, preds)  # Sensitivity = Recall
    cls_spec = calculate_specificity(y_test, preds)
    return cls_mcc, cls_accuracy, cls_precision, cls_sensitivity, cls_spec

def csv2fasta_func(csv_path, fasta_path):
    # get .csv info
    seq_data = pd.read_csv(csv_path)
    # .csv to .fasta
    fast_file = open(fasta_path, "w")
    for i in range(len(seq_data.SEQUENCE)):
        fast_file.write(">" + str(seq_data.name[i]) + "\n")
        fast_file.write(seq_data.SEQUENCE[i] + "\n")
    fast_file.close()

def aop_ml_run(data_name, data_split):
    df_path = '/home/jianxiu/OneDrive/aop/data/' + data_name + '/'
    my_seed = random.randint(0, 1000000)
    best_path = 'result/' + data_name + '_' + str(data_split) + ".csv"
    feature_name_list = ["DDE"] #,  "CKSAAGP","CTDC", "CTDT", "CTDD","DDE", "DPC", "GAAC", "GDPC", "GTPC"
    cls_name_list = ["SVM"] #,"SVM", "RF", "DT", "KNN", 'XGBoost'

    seed_list, encoding_list, cls_list, mcc_list, acc_list, precision_list, sensitivity_list, spec_list = [], [], [], [], [], [], [], []
    train_csv = df_path + str(data_split) + "/train.csv"
    test_csv = df_path + str(data_split)+  "/test.csv"

    train_fasta = df_path + str(data_split) + "/train.fasta"
    test_fasta = df_path + str(data_split) + "/test.fasta"

    csv2fasta_func(train_csv, train_fasta)
    csv2fasta_func(test_csv, test_fasta)

    train_df = pd.read_csv(train_csv)
    y_train = train_df["label"]
    test_df = pd.read_csv(test_csv)
    y_test = test_df["label"]
    for feature_name in feature_name_list:
        train_ft_df = GeneIfeature(train_fasta, ft_name=feature_name)
        test_ft_df = GeneIfeature(test_fasta, ft_name=feature_name)
        train_ft_path = "pep_features/" + data_name + "_train_" + feature_name + ".pkl"
        test_ft_path = "pep_features/"  + data_name + "_test_" + feature_name + ".pkl"

        train_ft_df.to_pickle(train_ft_path)
        test_ft_df.to_pickle(test_ft_path)
        #
        # train_ft = pd.read_pickle(train_ft_path)
        # test_ft = pd.read_pickle(test_ft_path)
        for cls_name in cls_name_list:
            cls_mcc, cls_accuracy, cls_precision, cls_sensitivity, cls_specificity  = ml_train_test(train_ft_df, y_train, test_ft_df, y_test, cls_name)
            seed_list.append(my_seed)
            encoding_list.append(feature_name)
            cls_list.append(cls_name)

            mcc_list.append(cls_mcc)
            acc_list.append(cls_accuracy)
            precision_list.append(cls_precision)
            sensitivity_list.append(cls_sensitivity)
            spec_list.append(cls_specificity)
    print("smart")
    ml_result_dict = {'seed': seed_list, "encoding": encoding_list, "cls": cls_list,
                        "mcc": mcc_list, "acc": acc_list, "precision":precision_list, "sensitivity":sensitivity_list, "specificity":spec_list}
    ml_result_df = pd.DataFrame(ml_result_dict)
    ml_result_df.to_csv(best_path)

if __name__ == "__main__":
    # 'HC100',
    data_str_list = ['AnOxPePred'] # , 'AnOxPP', 'AOPP'
    for i in range(5):
        for data_name in data_str_list:
            aop_ml_run(data_name, i)
    print("smart")
