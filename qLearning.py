#!env python3
"""
# Our Q learner applied on 421

Group: DE POORTER Marianne

"""
import random
import json
import matplotlib.pyplot

# MAIN:

def main():
    import game421 as game

    gameEngine = game.Engine()

    print("You will be asked for epsilon, alpha and gamma.\nIf one of them is NaN, all values are set to default.\n")
    epsilon = input('Please enter the epsilon value, default is 0.1.\n')
    gamma = input('Please enter the gamma value, default is 0.99.\n')
    alpha = input('Please enter the alpha value, default is 0.1.\n')
    print("\r--> Training a player from the beginning...")

    if epsilon is not None and gamma is not None and alpha is not None:
        try:
            epsilon = float(epsilon)
            gamma = float(gamma)
            alpha = float(alpha)
            player = PlayerQMDP(epsilon=epsilon, alpha=alpha, gamma=gamma)
        except ValueError:
            print("Applying default values...")
            player = PlayerQMDP()
        except TypeError:
            print("Applying default values...")
            player = PlayerQMDP()
    else:
        player = PlayerQMDP()

    numberOfGames = 100
    big_average = 0  # average over the 1000 episodes
    max_average = 0  # maximum average hit on one episode (100 games of 2 rounds)

    for episode in range(1000):
        rewards = gameEngine.start(player, numberOfGames)
        average = sum(rewards) / len(rewards)
        big_average += average
        if average > max_average:
            max_average = average

    # sets of debug tools to show progress
    """print("Average: " + str(average))
    print("Q value for initialisation : ", player.qvalues['9-1-1-1'])
    print("Number of non-zero values : ", player.countNotNull())
    print("Exploration count : ", player.exploration_count)
    print("Exploitation count : ", player.exploitation_count)"""

    print("Total average (1000 episodes of 100 games of 2 rounds) : ",
          str(big_average / 1000))  # printing the overall average
    print("Maximum average (100 games of 2 rounds) : ", max_average)

    # Serializing json
    json_object = json.dumps(player.qvalues, indent=4)
    # Eventually printing it
    # print(json_object)

    # Dumping json into a file
    with open("trained_qvalues.json", "w") as outfile:
        json.dump(player.qvalues, outfile)

    # Initialize the q_values with the file given
    print("\r--> Using the pre-trained q-values from the file we saved...")
    # Opening JSON file
    with open('trained_qvalues.json') as json_file:
        data = json.load(json_file)

    if epsilon is not None and gamma is not None and alpha is not None:
        try:
            epsilon = float(epsilon)
            gamma = float(gamma)
            alpha = float(alpha)
            player_pre_trained = PlayerQMDP(epsilon=epsilon, alpha=alpha, gamma=gamma, qvalues=data)
        except ValueError:
            print("Applying default values...")
            player_pre_trained = PlayerQMDP(qvalues=data)
        except TypeError:
            print("Applying default values...")
            player_pre_trained = PlayerQMDP(qvalues=data)
    else:
        player_pre_trained = PlayerQMDP(qvalues=data)

    big_average = 0  # average over the 1000 episodes
    max_average = 0  # maximum score hit for one play (two rounds)

    for episode in range(1000):
        rewards = gameEngine.start(player_pre_trained, numberOfGames)
        average = sum(rewards) / len(rewards)
        big_average += average
        if average > max_average:
            max_average = average
        # print("Average: " + str(average))
    print("Total average (1000 episodes of 100 games of 2 rounds) : ",
          str(big_average / 1000))  # printing the overall average
    print("Maximum average (100 games of 2 rounds) : ", max_average)

    # Dumping json into a file
    with open("re_trained_qvalues.json", "w") as outfile:
        json.dump(player_pre_trained.qvalues, outfile)

    # Using a PlayerBestQMDP where we only choose the best options from a given set of qvalues
    print("\r--> Using a pre-trained q-values set on a player in best-only mode...")
    with open('re_trained_qvalues.json') as json_file:
        new_data = json.load(json_file)
    best_player = PlayerBestQMDP(qvalues=new_data)
    big_average = 0.0
    max_average = 0.0
    for episode in range(10):
        rewards = gameEngine.start(best_player, numberOfGames)
        average = sum(rewards) / len(rewards)
        big_average += average
        if average > max_average:
            max_average = average
    print("Total average (10 episodes of 100 games of 2 rounds) : ", str(big_average / 10))
    print("Maximum average (100 games of 2 rounds) : ", max_average)


# ACTIONS:

actions = []
for a1 in ['keep', 'roll']:
    for a2 in ['keep', 'roll']:
        for a3 in ['keep', 'roll']:
            actions.append(a1 + '-' + a2 + '-' + a3)


def plot_qvalues(data):
    return None


# Q LEARNER:

class PlayerQMDP:
    init_state = '9-1-1-1'

    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.99, qvalues=None):
        if qvalues is None:
            qvalues = {}
        self.results = []
        self.qvalues = qvalues
        self.exploration_count = 0
        self.exploitation_count = 0
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        print("Epsilon: ", self.epsilon, "; gamma: ", self.gamma, "; alpha :", self.alpha)

    # State Machine :
    def stateStr(self):
        s = str(self.turn)
        for d in self.dices:
            s += '-' + str(d)
        return s

    # AI interface :
    def wakeUp(self, numberOfPlayers, playerId, tabletop):
        # print("\nNew Game !! \r")

        self.scores = [0 for i in range(numberOfPlayers)]
        self.id = playerId
        self.model = tabletop
        self.turn = 9
        self.dices = [1, 1, 1]
        self.action = 'roll-roll-roll'

        state = self.stateStr()
        if state not in self.qvalues.keys():
            self.qvalues[state] = {"keep-keep-keep": 0.0, "roll-keep-keep": 0.0, "keep-roll-keep": 0.0,
                                   "roll-roll-keep": 0.0, "keep-keep-roll": 0.0, "roll-keep-roll": 0.0,
                                   "keep-roll-roll": 0.0, "roll-roll-roll": 0.0}

    def perceive(self, turn, scores, pieces):
        last = self.stateStr()

        self.reward = scores[self.id] - self.scores[self.id]
        self.scores = scores
        self.turn = turn
        self.dices = pieces

        state = self.stateStr()

        if state not in self.qvalues.keys():
            self.qvalues[state] = {"keep-keep-keep": 0.0, "roll-keep-keep": 0.0, "keep-roll-keep": 0.0,
                                   "roll-roll-keep": 0.0, "keep-keep-roll": 0.0, "roll-keep-roll": 0.0,
                                   "keep-roll-roll": 0.0, "roll-roll-roll": 0.0}
        if state in self.qvalues.keys():
            q = (1 - self.alpha) * self.qvalues[last][self.action] + self.alpha * (
                    self.reward + self.gamma * self.findMax(state)[0])
            self.qvalues[last][self.action] = q

    def decide(self):
        mode = random.randrange(0, 10) / 10
        if self.stateStr() == self.init_state or mode <= self.epsilon:
            self.action = random.choice(actions)
            self.exploration_count += 1
        else:
            # find best action
            self.action = self.findMax(self.stateStr())[1]
            self.exploitation_count += 1
        # print(f'state: {self.stateStr()}, action: {self.action}')
        return self.action

    def sleep(self, result):
        self.results.append(result)

    def findMax(self, state):
        max_value = 0.0
        best_action = random.choice(actions)
        # print("values : ", self.qvalues[state])
        for action in self.qvalues[state].keys():
            if self.qvalues[state][action] > max_value:
                max_value = self.qvalues[state][action]
                best_action = action
        #  print("best action :", best_action, "; best value :", max_value)
        return max_value, best_action

    def countNotNull(self):
        count = 0.0
        for state in self.qvalues.keys():
            for action in self.qvalues[state].keys():
                if self.qvalues[state][action] != 0.0:
                    count += 1
        return count


# BEST Q-LEARNER

class PlayerBestQMDP:
    init_state = '9-1-1-1'

    def __init__(self, qvalues):
        self.results = []
        self.qvalues = qvalues
        self.exploration_count = 0
        self.exploitation_count = 0

    # State Machine :
    def stateStr(self):
        s = str(self.turn)
        for d in self.dices:
            s += '-' + str(d)
        return s

    # AI interface :
    def wakeUp(self, numberOfPlayers, playerId, tabletop):
        # print("\nNew Game !! \r")

        self.scores = [0 for i in range(numberOfPlayers)]
        self.id = playerId
        self.model = tabletop
        self.turn = 9
        self.dices = [1, 1, 1]
        self.action = 'roll-roll-roll'

    def perceive(self, turn, scores, pieces):
        last = self.stateStr()

        self.reward = scores[self.id] - self.scores[self.id]
        self.scores = scores
        self.turn = turn
        self.dices = pieces

    def decide(self):
        # find best action
        self.action = self.findMax(self.stateStr())[1]
        self.exploitation_count += 1
        # print(f'state: {self.stateStr()}, action: {self.action}')
        return self.action

    def sleep(self, result):
        self.results.append(result)

    def findMax(self, state):
        max_value = 0.0
        best_action = random.choice(actions)
        # print("values : ", self.qvalues[state])
        for action in self.qvalues[state].keys():
            if self.qvalues[state][action] > max_value:
                max_value = self.qvalues[state][action]
                best_action = action
        #  print("best action :", best_action, "; best value :", max_value)
        return max_value, best_action

    def countNotNull(self):
        count = 0.0
        for state in self.qvalues.keys():
            for action in self.qvalues[state].keys():
                if self.qvalues[state][action] != 0.0:
                    count += 1
        return count


# SCRIPT EXECUTION:
if __name__ == '__main__':
    print("Let's go !!!")
    main()
