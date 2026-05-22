import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Define the discrete value range of the feature quantity

file_path1 = r"C:\\Users\\60171\\Desktop\\_60\\ratio.xlsx"
df = pd.read_excel(file_path1,sheet_name="mnew")
nihss = df.iloc[:,1].to_numpy()
# print(f"nihss:{nihss}")
mrs = df.iloc[:,2].to_numpy()
# print(f"mrs:{mrs}")
# print(df.iloc[:,1])
mrs[mrs <= 2] = 1
mrs[mrs > 2] = 0
data = df.iloc[:, 3:].to_numpy() * 100
name = df.iloc[:,0]
# print(name.size)
# x=[8,8,3,29,2,17,23,11,24,2]
# tmp = np.array(x)
# tmp = np.hstack((tmp[:], tmp[:]))
# penalties = np.sum(data > tmp, axis=1)
# scores = 10 - penalties
# df = pd.DataFrame(scores)

# Write to Excel file
# df.to_excel(r"C:\Users\60171\Desktop\ouput.xlsx")
# Define the objective function
def objective_functions(x,idx,fold=0):
    # Here, the objective function should be defined based on the specific practical context
    # Below is an example objective function. It should be replaced with the actual objective function as needed
    tmp = np.array(x)
    tmp = np.hstack((tmp[:], tmp[:]))
    penalties = np.sum(data > tmp, axis=1)
    scores = 10 - penalties
    f1, _ = spearmanr(scores[idx], nihss[idx])
    # print(f"thresholds：{tmp},f1: {f1}")
    f2 = -roc_auc_score(mrs[idx], scores[idx]/10)
    # print(name[idx].shape,mrs[idx].shape,scores[idx].shape)
    if fold!=0:
        df2=pd.DataFrame({'Name': name[idx],
                        'MRS': mrs[idx],
                        'Scores': scores[idx]})
        
        df2.to_excel(f"C:\\Users\\60171\\Desktop\\_{fold}.xlsx")
    return [f1, f2]

# Define the multi-objective optimization problem
class DiscreteMultiObjectiveProblem(ElementwiseProblem):
    def __init__(self, idx):
        super().__init__(n_var=10,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([0] * 10),
                         xu=np.array([30] * 10),
                         vtype=int)
        self.idx = idx

    def _evaluate(self, x, out, *args, **kwargs):
        # Handle the discrete values
        # thresholds = [feature_ranges[i][np.argmin(np.abs(feature_ranges[i] - x[i]))] for i in range(10)]

        # calculate score
        f1,f2 = objective_functions(x,self.idx)
        out["F"] = [f1, f2]

# Run cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
train = []
test = []
# Initialize the optimization algorithm
algorithm = NSGA2(pop_size=100,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.9, eta=15,vtype=int),
    mutation=PM(eta=20,vtype=int),
    eliminate_duplicates=True)
fold=0
for train_index, test_index in kf.split(data):  # A genetic algorithm optimization is performed for each fold
    # train.append(train_index)
    # test.append(test_index)
    # The evaluation function is called with train_index and test_index, ensuring that each fold receives separate and independent training and testing subsets
    # fold=fold+1
    # if fold==1:
        # pd2=pd.DataFrame(name[test_index])
        # pd2.to_excel(r"C:\Users\60171\Desktop\name.xlsx")

    problem = DiscreteMultiObjectiveProblem(train_index)
    # Define a problem instance
    res = minimize(problem,
               algorithm,
               ('n_gen', 400),
               seed=1,
               verbose=False)
    F = np.array([solution.F for solution in res.opt])

    # Step 1: Find the solution that minimizes objective 1
    min_f1 = np.min(F[:, 0])
    candidates = F[F[:, 0] == min_f1]

    # Step 2: From the candidate solutions, select the one that minimizes objective 2
    best_idx_in_candidates = np.argmin(candidates[:, 1])
    best_solution = res.opt[np.where((F[:, 0] == min_f1) & (F[:, 1] == candidates[best_idx_in_candidates, 1]))[0][0]]

    fold=fold+1
    print("Select the solution according to priority:", best_solution.X)
    f1, f2 = objective_functions(best_solution.X,train_index,fold)
    print(f"fold: {fold}, Objective function value on the training set: f1 = {f1}, f2 = {f2}")

    # f1, f2 = objective_functions(best_solution.X,test_index,fold)
    # print(f"Objective function value on the test set: f1 = {f1}, f2 = {f2}")
    # Print the results for each fold
    # print("Pareto Optimal solution:")
    # for thresholds in res.X:
    #     # thresholds = [feature_ranges[i][np.argmin(np.abs(feature_ranges[i] - thresholds[i]))] for i in range(10)]
    #     f1, f2 = objective_functions(thresholds,train_index)
    #     print(f"thresholds: {thresholds}, Objective function value on the training set: f1 = {f1}, f2 = {f2}")
    #     f1, f2 = objective_functions(thresholds,test_index)
    #     print(f"Objective function value on the test set: f1 = {f1}, f2 = {f2}")
