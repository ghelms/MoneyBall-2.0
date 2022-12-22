#%%
import pulp
import pandas as pd

df = pd.read_csv("data/player_stats.csv")
df[df["yearID"] == 2020].head()
df = df[["playerID", "br_WAR_total", "salary"]]
# get rows where no column has NaN
df = df.dropna()
# keep unique playerID and keep the first if duplicates are found
df = df.drop_duplicates(subset="playerID", keep="first")
df1 = df.iloc[:5]

import numpy as np
from scipy.optimize import minimize_kkt
#%%
# Define the objective function
def objective(x):
    # x is an array of decision variables (in this case, the players on the team)
    # Calculate the total number of wins for the team using the decision variables
    total_wins = 5 * x[0] + 2 * x[1]
    return total_wins

# Define the inequality constraint (maximum budget for the team)
def budget_constraint(x):
    # x is an array of decision variables (in this case, the players on the team)
    # Calculate the total cost of the team using the decision variables
    total_cost = x[0] * 100 + x[1] * 50
    return total_cost - 1000

# Define the equality constraint (maximum team size)
def team_size_constraint(x):
    # x is an array of decision variables (in this case, the players on the team)
    return x[0] + x[1] - 25

#%%
# Solve the optimization problem using the minimize_kkt() function
x0 = np.array([10, 15])  # Initial guess for the decision variables
constraints = [{'type': 'ineq', 'fun': budget_constraint},
               {'type': 'eq', 'fun': team_size_constraint}]
result = minimize_kkt(objective, x0,  constraints=constraints, data=None, options=None)

# Print the optimal values of the decision variables and the optimal value of the objective function
print(result.x)
print(result.fun)