import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from handyspark import *


def resample(df, 
             multiplier_minority, 
             fraction_majority, 
             label_col='label',
             minority_tag=1):
    '''
    Resample the binary imbalanced dataset 
    -- upsample the minority group and downsample the majority group

    Downsampling is more straightforward,  

    df: the binary imalanced dataset; 
    multiplier_minority: how many small proportions do we want of the minority data
    fraction_majority: sample fraction for majority group
    label_col: column name of the label column
    minority_tag: tag that has very few representatives
    ''' 
    po = df.filter("{} == {}".format(label_col, minority_tag))
    ne = df.filter("{} != {}".format(label_col, minority_tag))
    po_resample = po.sample(True, multiplier_minority+0.01)
    ne_resample = ne.sample(fraction=fraction_majority-0.01)
    return po_resample.union(ne_resample)
  

def display_roc(predictions, labelCol='label'):
    '''
    Display the ROC curve
    
    predictions: the prediction dataframe from the `transform` method of the pyspark.ml estimators
    '''
    bcm = BinaryClassificationMetrics(predictions, scoreCol='probability', labelCol=labelCol)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    bcm.plot_roc_curve(ax=axs[0])
    bcm.plot_pr_curve(ax=axs[1])
    display(plt.show())


def generate_train_test(data,
                        multiplier_minority, 
                        fraction_majority, 
                        label_col='label',
                        minority_tag=1,
                        train_perc=0.7):
    '''
    Train test split on the data (after the step of features assembling)

    multiplier_minority: how many small proportions do we want of the minority data
    fraction_majority: sample fraction for majority group
    label_col: column name of the label column
    minority_tag: tag that has very few representatives
    train_perc: how many percentages of the data will go to the training set
    '''
    po = data.filter("{} == {}".format(label_col, minority_tag))
    ne = data.filter("{} != {}".format(label_col, minority_tag))
    training_po, testing_po = po.randomSplit([train_perc, 1-train_perc], seed = 100)
    training_ne, testing_ne = ne.randomSplit([train_perc, 1-train_perc], seed = 100)
    training = training_po.union(training_ne)
    training = resample(training, 
                        multiplier_minority=multiplier_minority, 
                        fraction_majority=fraction_majority)
    testing = testing_po.union(testing_ne)
    scaler = StandardScaler(inputCol="features", 
                            outputCol="scaledFeatures",
                            withStd=True, 
                            withMean=False)
    scale_model = scaler.fit(training)
    training = scale_model.transform(training)
    testing = scale_model.transform(testing)
    return training, testing


def tune_sampling_parameters(data,
                            estimator,
                            multiplier_minority,
                            fraction_majority,
                            num_folds=5):
    '''
    This is a helper function for tuning sampling parameter;
    it implements train test split,  training data resampling, model fitting 
    and evaluation for `num_folds` times and returns a list of auc scores of 
    the `num_folds` models so that we can evaluate the model in a more 
    robust way.

    data: the binary imalanced dataset; 
    estimator: pyspark ml estimator;
    multiplier_minority: how many small proportions do we want of the minority data
    fraction_majority: sample fraction for majority group
    num_folds: numbers of folds
    '''
    auc_list = []
    count_po = data.filter("label == 1").count() * 0.7
    count_ne = data.filter("label == 0").count() * 0.7
    print("Minority -- training size went from {} to {}".format(count_po, count_po * multiplier_minority))
    print("Majority -- training size went from {} to {}".format(count_ne, count_ne * fraction_majority))

    for i in range(num_folds):
        print("fold {}".format(i))
        training, testing = generate_train_test(data, 
                                                multiplier_minority, 
                                                fraction_majority)
        print(training.filter("label==1").count())
        print(training.filter("label==0").count())
        model = estimator.fit(training)
        predictions = model.transform(testing)
        bcm = BinaryClassificationMetrics(predictions, scoreCol='probability', labelCol='label')
        auc_list.append(bcm.areaUnderROC)
    return auc_list


def tune_model_parameter(data,
                         multiplier_minority, 
                         fraction_majority,
                         estimator,
                         paramgrid,
                         numFolds=3):
    '''
    User CrossValidator to tune the parameters of the model(estimator)

    multiplier_minority: how many small proportions do we want of the minority data
    fraction_majority: sample fraction for majority group
    estimator: a pyspark.ml estimator
    paramgrid: the parameters grid built by ParamGridBuilder
    numFolds: numbers of validation folds
    '''
    
    training, testing = generate_train_test(data, 
                                            multiplier_minority, 
                                            fraction_majority)
    crossval = CrossValidator(estimator=estimator,
                              estimatorParamMaps=paramgrid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=2)
    cvModel = crossval.fit(training)
    predictions = cvModel.transform(testing)
    display_roc(predictions)
    return cvModel
    

def plot_important_features(model,
                            feature_names):
    '''
    Display the bar chart that shows the feature importance scores of the most important 10 features

    model: the fitted model
    feature_names: feature names from assembler.getInputCols()
    '''
    feature_importance = pd.DataFrame({'features':feature_names, 'scores':model.featureImportances})
    plt.figure(figsize=(20,5))
    top_featureas = feature_importance.sort_values('scores', ascending=False).head(10)
    plt.bar(top_featureas['features'], top_featureas['scores'])
    plt.title('Important features')
    display(plt.show())

