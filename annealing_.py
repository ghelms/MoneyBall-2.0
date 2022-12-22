import pandas as pd

df = pd.read_csv("data/player_stats-2.csv")
#df = df[df["yearID"]]
df = df[df["leagueID"] == "NL"]
df = df[["playerID", "br_WAR_total", "salary"]]
# get rows where no column has NaN
df = df.dropna()
## keep unique playerID and keep the first if duplicates are found
df = df.drop_duplicates(subset="playerID", keep="first")

import numpy as np
# uniformly sample 1000000 values from the range of 0 to 1
gen_war_scores = np.random.uniform(min(df["br_WAR_total"]) - 10, max(df["br_WAR_total"])+10, 10000)
gen_salaries = np.random.uniform(min(df["salary"]), max(df["salary"])+1000000, 10000)

# create a dataframe with the generated values
gen_df = pd.DataFrame({"br_WAR_total": gen_war_scores, "salary": gen_salaries})
#give each row a unique id
gen_df["playerID"] = gen_df.index

BUDGET = 50_000_000
SAMPLE_SIZE = 10000
PLAYER_COUNT = 26
BOOTSTRAP_SIZE = 100
'''

import pulp

df1 = gen_df

# Create a linear programming model
model = pulp.LpProblem("Best Baseball Team", pulp.LpMaximize)

# Define the variables for the problem
# Each variable represents a player and has a value of 1 if the player is selected for the team and 0 otherwise
player_vars = pulp.LpVariable.dicts("Player", df1["playerID"].unique().tolist(), 0, 1, pulp.LpInteger)

# Define the objective function
# The objective function is to maximize the total WAR of the team
model += sum(player_vars[row["playerID"]] * row["br_WAR_total"] for index, row in df1.iterrows())

# Define the budget constraint
# The total salary of the team must be less than or equal to the budget
model += sum(player_vars[row["playerID"]] * row["salary"] for index, row in df1.iterrows()) <= BUDGET

# Define the position constraints
# The number of players at each position must be within the allowed range
model += sum(player_vars[row["playerID"]] for index, row in df1.iterrows()) == 26

# Solve the optimization problem
model.solve()
ground_truth = 0
for var in model.variables():
    if var.varValue == 1:
        print(var.name)
        #find the row where playerID is equal to the variable name
        ground_truth += df1[df1["playerID"] == int(var.name[7:])]["br_WAR_total"].values[0]
print(ground_truth)

'''
import random
import math
import numpy as np

def simulated_annealing(df, numIterations, budget):
  df = df.sort_values(by="salary", ascending=True)
  playerSalaries, playerPerformance = df["salary"].tolist(), df["br_WAR_total"].tolist()
  print("hello")
  # Initialize the current team to a randomly selected group of 26 players
  currentCost =  budget + 1
  
  while currentCost > budget:
    currentTeam = list(range(26))#list(np.random.choice(list(range(len(playerSalaries))), 26, replace=False))
    currentPerformance = sum([playerPerformance[i] for i in currentTeam])
    currentCost = sum([playerSalaries[i] for i in currentTeam])

  # Set the initial temperature
  temperature = 1000.0

  # Set the cooling rate
  coolingRate = 0.003

  # Initialize the best team and performance to the current team and performance
  bestTeam = currentTeam
  bestPerformance = currentPerformance

  # Iterate over the specified number of iterations
  for i in range(numIterations):
    # Generate a new team by randomly swapping out one player in the current team
    newTeam = currentTeam.copy()

    indexToChange = random.randint(0, 25)

    #get row equivalent player index
    newPlayerIndex = newTeam[indexToChange]


    while newPlayerIndex == newTeam[indexToChange] or newPlayerIndex in newTeam[:indexToChange] or newPlayerIndex in newTeam[indexToChange+1:]:
      newPlayerIndex = random.randint(0, len(playerSalaries) - 1)


    newTeam[indexToChange] = newPlayerIndex


    newPerformance = sum([playerPerformance[i] for i in newTeam])
    newCost = sum([playerSalaries[i] for i in newTeam])

    # Calculate the acceptance probability
    acceptanceProbability = calcAcceptanceProbability(currentPerformance, newPerformance, temperature)

    # Determine whether to accept the new team
    if (newCost <= budget and newPerformance > currentPerformance) or (newCost <= budget and random.random() < acceptanceProbability):
      currentTeam = newTeam
      currentPerformance = newPerformance
      currentCost = newCost

      # Update the best team and performance if necessary
      if newPerformance > bestPerformance:
        bestTeam = newTeam
        bestPerformance = newPerformance

    # Cool the temperature
    temperature *= 1 - coolingRate
  return sum(df.iloc[bestTeam]["br_WAR_total"])

def calcAcceptanceProbability(currentPerformance, newPerformance, temperature):
  if newPerformance > currentPerformance:
    return 1.0
  else:
    return math.exp((newPerformance - currentPerformance) / temperature)

#print("Uniform: ", sample_uniform(df, SAMPLE_SIZE, BUDGET))
#print("Oversample: ",sample_over(df, SAMPLE_SIZE, BUDGET))
print("Simulated Annealing: ",simulated_annealing(gen_df, SAMPLE_SIZE, BUDGET))
