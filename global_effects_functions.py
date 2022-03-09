
import pandas as pd
from lightgbm import LGBMClassifier
import numpy as np
import seaborn as sns
from scipy import stats
import random
import matplotlib.pyplot as plt


def global_effects_xy(dataframe: pd.DataFrame, n_vals: int, model):
    """Create global effects data for partial dependence plots"""
    x_val_dict = {}
    y_val_dict = {}

    for col in dataframe.columns:
        # get x
        min_val = train[col].min()
        max_val = train[col].max()
        col_vals = np.linspace(start=min_val, stop=max_val, num=n_vals)
        x_val_dict[col] = col_vals

        # get y
        new_data = dataframe.copy()
        new_data.loc[:, :] = 0
        new_data = new_data.head(n_vals)
        new_data[col] = x_val_dict[col]

        preds = model.predict(new_data, raw_score=True)
        preds_standardized = stats.zscore(preds)
        y_val_dict[col] = preds_standardized

    return x_val_dict, y_val_dict


def bootstrap_samples(n_boots: int, data: pd.DataFrame):

    boots = range(n_boots)
    boot_list = []

    for i in boots:
        boot_num = i
        sample = random.choices(range(len(df)), k=len(df))
        boot_list.append((boot_num, sample))
    return boot_list


def get_sampled_predictions(data: pd.DataFrame, target_var: str, n_samples: int, x_vals: dict, n_vals: int):

    sampling_data = data.copy()

    list_of_y_dicts = []

    for i in np.arange(0, n_samples):

        # sample data
        #bootstrap_list = bootstrap_samples(n_boots=n_samples, data=sampling_data)
        #indices = bootstrap_list[i][1]
        #sampled_data = sampling_data.iloc[indices]
        sampled_data = sampling_data.sample(frac=1, replace=True)
        target = sampled_data[target_var].values
        sampled_data = sampled_data.drop([target_var], axis=1)

        # train model
        model = LGBMClassifier(max_depth=1, n_estimators=100)
        model.fit(sampled_data, target)

        # predict on x values
        y_val_dict = {}
        for col in sampled_data.columns:
            new_data = data.copy()
            new_data = new_data.drop([target_var], axis=1)
            new_data.loc[:, :] = 0
            new_data = new_data.head(n_vals)
            new_data[col] = x_vals[col]
            preds = model.predict(new_data, raw_score=True)
            preds_standardized = stats.zscore(preds)
            y_val_dict[col] = preds_standardized

        #output and collect results
        list_of_y_dicts.append(y_val_dict)

    return list_of_y_dicts


def get_conf_intervals(list_of_pred_dicts: list, data: pd.DataFrame, n_samples: int, n_vals: int):

    vals_dict = {}
    for col in data.columns:
        col_list = []
        for x in np.arange(0, n_vals):
            x_list = []
            for i in list_of_pred_dicts:
                try:
                    val = i[col][x]
                    x_list.append(val)
                except:
                    pass
            col_list.append(x_list)

        vals_dict[col] = col_list

    # get CI
    ci_dict = {}
    for col in data.columns:
        val_list = []
        for sample_preds in vals_dict[col]:
            if len(set(sample_preds)) > 1:
                try:
                    ci = stats.t.interval(alpha=0.99,
                                          df=len(sample_preds) - 1,
                                          loc=np.mean(sample_preds),
                                          scale=stats.sem(sample_preds))

                except:
                    ci = (np.mean(sample_preds),np.mean(sample_preds))
            else:
                ci = (np.mean(sample_preds), np.mean(sample_preds))

            val_list.append(ci)
        ci_dict[col] = val_list

    return ci_dict


def weighted_average(distribution, weights):
    return round(sum([distribution[i]*weights[i] for i in range(len(distribution))])/sum(weights),2)


def line_with_dist_plot(x: np.ndarray, y: np.ndarray, ci_levels: list, x_vals: np.ndarray, var_name: str):
    """
    Plots a line plot, with uncertainty band. Furthermore plots a histogram of x-values on top of the line plot.
    """

    # Create joint grid:
    g = sns.JointGrid()
    # Remove marginal y plot
    g.ax_marg_y.remove()

    # line plot and confidence interval:
    g.ax_joint.plot(x, y, label='Prediction')

    ci_lower = [el[0] for el in ci_levels]
    ci_upper = [el[1] for el in ci_levels]

    g.ax_joint.fill_between(x=x, y1=ci_lower, y2=ci_upper, alpha=0.2, label="Prediction interval")

    # distribution plot:
    sns.histplot(x=x_vals, ax=g.ax_marg_x)

    # Remove axes, change fonts, add grid and text.
    g.ax_joint.spines['left'].set_visible(False)
    g.ax_joint.spines['bottom'].set_visible(False)
    plt.yticks(fontfamily='Open Sans', fontsize=14)
    plt.xticks(fontfamily='Open Sans', fontsize=14)
    g.ax_joint.grid(alpha=0.3)
    plt.suptitle(var_name, fontproperties={'family': 'Ubuntu', "size": 20})
    g.ax_joint.set_ylabel(var_name)

    plt.tight_layout()
    plt.show()

train = pd.read_csv("train.csv")
train_w_target =  train.drop(['PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin', 'Sex'], axis=1)
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin', 'Sex', 'Survived'], axis=1)
Target = train_w_target["Survived"].values
target_var = "Survived"

mod = LGBMClassifier(max_depth=1, n_estimators=100)
mod.fit(train, Target)

n_vals = 200
n_samples = 10
data = train
data_w_target = train_w_target

x, y = global_effects_xy(dataframe=data, n_vals=n_vals, model=mod)

list_of_pred_dicts = get_sampled_predictions(data=data_w_target, target_var=target_var, n_samples=n_samples, x_vals=x, n_vals=n_vals)

ci = get_conf_intervals(list_of_pred_dicts, data=data, n_samples=n_samples, n_vals=n_vals)


# columns
columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for col in columns:
    if col == "Pclass":
        x_plot = x[col]
        y_plot = y[col]
        x_vals = train[col].values

        # get weights to scale y
        x_vals_df = pd.DataFrame({"x_vals": x_vals})
        x_vals_df['bin'] = pd.cut(x_vals_df['x_vals'].values, bins=len(df))
        weights = x_vals_df.bin.value_counts().values
        weighted_avg = np.average(y_plot, weights=weights)
        y_plot = y_plot - weighted_avg
        ci_band = ci[col] - weighted_avg
        line_with_dist_plot(x=x_plot, y=y_plot, ci_levels=ci_band, x_vals=x_vals, var_name=col)



"""

d=get_conf_intervals(list_of_pred_dicts,data=train,n_samples=5, n_vals=100)

k=[0.3382827238757179, 1.3382827238757179, 1.3382827238757179, 2.3382827238757179, 1.3382827238757179]

stats.t.interval(alpha=0.95, df=len(k) - 1, loc=np.mean(k), scale=stats.sem(k))

# get confidence intervals

    # dict with column names as keys
    #  - list of length n_vals
    #     -  with n_samples
    vals_dict = {}

    for col in data.columns:
        col_list = []
        for x in xpos:
            x_list = []
            for i in pred_dict:
                try:
                    val = i[col][x]
                    x_list.append(val)
                except:
                    pass
            col_list.append(x_list)

        vals_dict[col] = col_list

mydict = confidence_intervals(data=train, target=Target, n_samples= 10, x_vals= x, n_vals= 100)




        list_of_y_dicts[i][col][-1]



# Pclass, Age, SibSp, Parch, Fare
col='Pclass'
line_with_dist_plot(x[col], y[col], x_vals=train[col])

k = confidence_intervals(data=train,target=Target, n_samples=5, x_vals = x, n_vals=100)


# dict with column names as keys
#  - list of length n_vals
#     -  with n_samples

# i.e. for each column we get y values for all x values

y_val_dict_list[0]['Age'][1]
values_dict = {} # length = n columns

outside_list = []  # length of x
inside_list = []  # length n_samples


for sample in y_val_dict_list:
    for col in sample:
        column_vals = []
        for i in sample[col]:
            column_vals.append(i)


y_val_dict_list
v = []
for col in train.columns:
    for dct in y_val_dict_list:
        v_list = (dct[col])



        for xpos in np.arange(0, 3):
            x_list.append(dct[xpos])


for col in train.columns:
    for xpos in np.arange(0, 3):
        for i in np.arange(0, len(y_val_dict_list)):
            print(y_val_dict_list[i][col][xpos])


for sample in y_val_dict_list:  # length n_samples
    for col in sample:
        for xpos in np.arange(0, 100): # number of x positions
            sample[col][xpos]
            for i in value:
                inside_list.append(value)
        outside_list.append(inside_list)
    values_dict[col] = outside_list



y_val_dict_list[0]['Age'][10]

for i in y_val_dict_list:
    for

xpos = np.arange(0,3)
samples = y_val_dict_list

mydict = {}
cols=list(train.columns)
for col in cols:
    big_list = []

    for x in xpos:
        small_list = []
        for i in samples:
            v = i['Age'][x]
            small_list.append(v)
        big_list.append(small_list)
    mydict[col] = big_list


dct = {}

for col in cols:
    big_list = []
    for x in xpos:
        x_list = []
        for i in samples:
            try:
                val = i[col][x]
                x_list.append(val)
            except:
                print(i)
        big_list.append(x_list)

    dct[col] = big_list

dct
"""

#mean_effect < - sum(predict_X0 * histogram) / sum(histogram)  # predlink weighted by histogram of x
#effect < - predict_X0 - mean_effect  # scale to mean effect at zero


x = x["Fare"]
y = y["Fare"]
x_vals

df = pd.DataFrame({'x':x,'y':y})

x_vals_df = pd.DataFrame({"x_vals": x_vals})

x_vals_df['bin'] = pd.cut(x_vals_df['x_vals'].values, bins=len(df))

weights = x_vals_df.bin.value_counts().values

weighted_avg = round(np.average( df['y'], weights = weights),2)

weighted_avg


