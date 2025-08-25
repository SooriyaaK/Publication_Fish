import csv
from scipy.stats import shapiro
from scipy.stats import levene
from scipy import stats
import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA

"""
TODO: test and fix before analysis
"""

class StatsAnalyser:

    def __init__(self):
        #  Extarcting data from "evaluation_loss.csv" file for cosine
        cosine = np.array([
        17.7061,
        20.0664,
        20.4256,
        18.58,
        20.3451,
        20.4634,
        18.8391,
        21.6425,
        20.3374,
        17.5323,
        20.3779,
        20.5711,
        17.6875,
        21.1727,
        20.5124,
        18.5495,
        20.3569,
        20.3484,
        21.0485,
        22.3074,
        21.9025,
        19.0856,
        22.6122,
        20.1927,
        17.693,
        20.7592,
        20.6431,
        17.8369,
        20.3674,
        21.3079,
        15.4266,
        17.1752,
        19.8003,
        15.6693,
        19.0305,
        19.4818,
        15.7398,
        19.719,
        19.0982,
        15.7467,
        17.8519,
        19.5881,
        15.7676,
        18.5017,
        19.7395,
        15.7221,
        17.5879,
        19.7928,
        17.8321,
        17.8594,
        20.5681,
        15.8187,
        19.9709,
        19.391,
        15.539,
        17.5388,
        23.3462,
        15.5731,
        17.791,
        18.6966,
        17.3366,
        20.2694,
        20.3154,
        18.5829,
        21.8205,
        20.3958,
        18.3517,
        22.8613,
        20.3113,
        18.033,
        21.8554,
        20.4705,
        18.321,
        23.2523,
        20.2865,
        18.1357,
        20.532,
        20.3747,
        23.5518,
        21.4778,
        20.5058,
        18.2012,
        23.3586,
        20.4309,
        17.1152,
        20.332,
        22.5743,
        17.1148,
        20.3344,
        23.4729


        ])

        #  Extarcting data from "evaluation_loss.csv" file for mse kikctime
        mse_kicktime = np.array([
        0.713472559,
        0.518657207,
        0.537905312,
        0.580135174,
        0.519395323,
        0.550727784,
        0.584567103,
        0.515012667,
        0.539201964,
        0.92247689,
        0.51948735,
        0.531042392,
        0.95034249,
        0.51742044,
        0.53440267,
        0.935954154,
        0.519320024,
        0.539224147,
        0.610134287,
        0.539780135,
        0.532796735,
        0.624669614,
        0.507279868,
        0.548609639,
        0.809071444,
        0.519675406,
        0.53512652,
        0.757445164,
        0.519560223,
        0.649594806,
        0.702161491,
        0.430644448,
        0.531427185,
        0.570306799,
        0.427235679,
        0.513142919,
        0.605878438,
        0.421794567,
        0.507436202,
        0.584512443,
        0.430980824,
        0.522506459,
        0.628343224,
        0.42949832,
        0.521691213,
        0.64426456,
        0.431154119,
        0.527292228,
        0.555865105,
        0.441689757,
        0.519936588,
        0.638178059,
        0.416365763,
        0.512016779,
        0.552006496,
        0.431551124,
        0.733641392,
        0.612129543,
        0.43181522,
        0.503969413,
        0.924854593,
        0.530367174,
        0.545649306,
        0.58067215,
        0.5263929,
        0.54150581,
        0.839819359,
        0.513733899,
        0.548428628,
        0.890033395,
        0.528959314,
        0.587069033,
        0.887952877,
        0.52300661,
        0.548105105,
        0.83264857,
        0.530863029,
        0.544726881,
        0.609613126,
        0.543447126,
        0.54776824,
        0.805509452,
        0.502500466,
        0.546726347,
        0.70197824,
        0.531012928,
        0.698975307,
        0.721152321,
        0.530793466,
        0.734908885



        ])


        #  Extarcting data from "evaluation_loss.csv" file for mse trajectory
        mse_trajectory = np.array([
        271.4485615,
        5.47352068,
        1.110565243,
        5.263532269,
        3.128242291,
        1.108199767,
        4.212893551,
        8.368645286,
        1.111679234,
        21.27619192,
        2.963023254,
        1.106485851,
        17.04574108,
        2.549535023,
        1.113304116,
        5.682385009,
        3.24428457,
        1.109327171,
        5.748912338,
        5.240349161,
        1.230415382,
        2.481771044,
        14.46088104,
        1.126585428,
        181.4363629,
        2.770929474,
        1.176839837,
        285.8312319,
        3.534801668,
        1.373399649,
        19.48893925,
        4.168351568,
        1.095369459,
        9.122524651,
        6.178665366,
        1.058160495,
        7.907282832,
        9.084370076,
        1.070862893,
        7.069257342, 
        3.681861954,
        1.099378247,
        8.871513522,
        6.168589038, 
        1.095523895,
        7.152129698,
        3.007254102,
        1.087616533,
        5.431645184,
        3.251617288,
        1.553959167,
        7.364690458,
        11.63456363,
        1.067199854,
        44.22257547,
        2.767151806,
        1.576581209,
        86.96305758,
        2.55181267,
        1.350954911,
        7.905986796,
        1.448177203,
        1.122408287,
        3.873277835,
        9.002413047,
        1.127648652,
        16.94815344,
        18.83754524,
        1.131189918,
        12.05934691,
        9.471445412,
        1.198373536,
        11.97059822,
        16.84708164,
        1.146917574,
        18.78385228,
        2.47434707,
        1.121753467,
        10.05037533,
        1.59336473,
        1.592889602,
        17.72007799,
        22.90274329,
        1.128859105,
        69.56268894,
        1.360543615,
        1.402508234,
        95.14698403,
        1.267276303,
        1.537940991])

    def perform_shapiro_wilks_test(self, data):
        # Shapiro Wilk Test
        res = shapiro(data)
        p = res.statistic
        print(f'Shapiro WIlk Test {p}')

    def print_levene(self, data):
        levene_cosine_orientation = [data[:30], data[30:60], data[60:]]
        print(levene_cosine_orientation)

    def perform_levene(self, data):
        group1 = data[:30]
        group2 = data[30:60]
        group3 = data[60:]

        stat, p_value = levene(group1, group2, group3)
        print(f"Levene’s test statistic = {stat:.5f}, p-value = {p_value:.5f}")

    def perform_anova(self):
        # Load the data from the evaluation.csv file
        filename = "evaluation_loss.csv"

        with open(filename, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            raw_data = []
            for row in reader:
                raw_data.append(row)

        # extract Loss function and its respective data
        loss_function_to_use = "Cosine"

        reshaped_data = []
        for row in raw_data:
            reshaped_row = {}
            reshaped_row["fitness"] = (float(row[loss_function_to_use]) * (1))
            reshaped_row["bias_mode"] = row["Bias_Mode"]
            reshaped_row["sort_criteria"] = row["Sorted Velocity Criteria"]
            reshaped_row["model"] = row["Model"]
            reshaped_data.append(reshaped_row)

        # Create a dictionary for statsmodels 
        fitness = []
        bias_mode = []
        sort_criteria = []

        for row in reshaped_data:
            fitness.append(row["fitness"])
            bias_mode.append(row["bias_mode"])
            sort_criteria.append(row["sort_criteria"])


        data_dict = {
            "Fitness": fitness,
            "BiasMode": bias_mode,
            "SortCriteria": sort_criteria
        }

        # ANOVA
        model = ols('Fitness ~ C(BiasMode) + C(SortCriteria) + C(BiasMode):C(SortCriteria)', data=data_dict).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

    def print_anova_table(self):
        # load the data from the evaluation.csv file
        filename = "evaluation_loss.csv"

        with open(filename, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            raw_data = []
            for row in reader:
                raw_data.append(row)

        # Loss  - respective data
        loss_function_to_use = "MSE_Trajectory"

        reshaped_data = []
        for row in raw_data:
            reshaped_row = {}
            reshaped_row["fitness"] = float(row[loss_function_to_use])
            reshaped_row["bias_mode"] = row["Bias_Mode"]
            reshaped_row["sort_criteria"] = row["Sorted Velocity Criteria"]
            reshaped_row["model"] = row["Model"]
            reshaped_data.append(reshaped_row)

        fitness = []
        bias_mode = []
        sort_criteria = []

        for row in reshaped_data:
            fitness.append(row["fitness"])
            bias_mode.append(row["bias_mode"])
            sort_criteria.append(row["sort_criteria"])

        # Create a dictionary for statsmodels
        data_dict = {
            "Fitness": fitness,
            "BiasMode": bias_mode,
            "SortCriteria": sort_criteria
        }

        # ANOVA
        model = ols('Fitness ~ C(BiasMode) + C(SortCriteria) + C(BiasMode):C(SortCriteria)', data=data_dict).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)


        print(anova_table)

    def perform_tukey_test(self, fitness, sort_criteria):
        tukey = pairwise_tukeyhsd(endog=fitness, groups=sort_criteria, alpha=0.05)
        print(tukey)

    def perform_pairwise_tukey_test(self, fitness, sort_criteria):
        key = pairwise_tukeyhsd(endog=fitness, groups=sort_criteria, alpha=0.05)
        print(key)

    def compute_fitness_mean(self, data_dict):
        sum_distance = 0
        count_distance = 0

        for fitness, sort_criteria in zip(data_dict['Fitness'], data_dict['SortCriteria']):
            if sort_criteria == 'distance':
                sum_distance += fitness
                count_distance += 1

        mean_distance = sum_distance / count_distance

        print(f'Distance mean: {mean_distance}')

    def perform_posthoc_test(self):
        # Load the data from the evaluation.csv file
        filename = "evaluation_loss.csv"
        with open(filename, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            raw_data = []
            for row in reader:
                raw_data.append(row)

        loss_function = "Cosine" 
        sort_method = "distance"

        # Filter data: Sorted Velocity Criteria = distance
        fitness = []
        bias_mode = []

        for row in raw_data:
            if row["Sorted Velocity Criteria"].strip().lower() == sort_method:
                fitness_val = float(row[loss_function])
                fitness.append(fitness_val)
                bias_mode.append(row["Bias_Mode"])

        # ANOVA
        data_dict = {
            "Fitness": fitness,
            "BiasMode": bias_mode
        }

        model = ols("Fitness ~ C(BiasMode)", data=data_dict).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print(f"\n=== ANOVA for BiasMode (Loss = {loss_function}, Sort = {sort_method}) ===")
        print(anova_table)

        # Tukey HSD post-hoc test
        print(f"\n=== Tukey HSD for BiasMode (Loss = {loss_function}, Sort = {sort_method}) ===")
        tukey = pairwise_tukeyhsd(endog=fitness, groups=bias_mode, alpha=0.05)
        print(tukey)

    def perform_ttest(self, mse_kicktime):
        # for RQ1
        results = stats.ttest_1samp(
            mse_kicktime, 
            popmean=0,
            alternative = 'greater'
        )

        print(f"t({results.df}) = {results.statistic:.2f}, p = {results.pvalue:.4f}")

    def print_pc_values_for_weights(self, results):
        weights = []
        runs = []
        for result in results:
            weights.append(result[3:])
            runs.append(f'R{result[1]}')

        weights_np = np.array(weights)

        # 1. Standardize
        scaled = StandardScaler().fit_transform(weights_np)

        # 2. PCA
        pca = PCA(n_components=2)
        scores = pca.fit_transform(scaled) 

        # 3. Extract loadings
        loadings = pca.components_ 

        df_loadings = pd.DataFrame(
            loadings,
            index=['PC1','PC2'],
            columns=['w1','w2','w3','w4','w5','bias']
        )
        print(df_loadings)