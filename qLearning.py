#!env python3
"""
# Our Q learner applied on 421

Group: NOM Prenom, NOM Prenom

"""
import random


# MAIN:

def main():
    import game421 as game

    gameEngine = game.Engine()
    player = PlayerQMDP()
    numberOfGames = 100
    big_average = 0
    max_score = 0

    for episod in range(1000):
        rewards = gameEngine.start(player, numberOfGames)
        average = sum(rewards) / len(rewards)
        big_average += average
        if average > max_score:
            max_score = average
        """print("Average: " + str(average))
        print("Q value for initialisation : ", player.qvalues['9-1-1-1'])
        print("Number of non-zero values : ", player.countNotNull())
        print("Exploration count : ", player.exploration_count)
        print("Exploitation count : ", player.exploitation_count)"""
    print("Total average : ", str(big_average / 1000))
    print("Maximum score : ", max_score)

    player.epsilon = 0.0

    big_average = 0.0
    max_score = 0.0
    for episod in range(10):
        rewards = gameEngine.start(player, numberOfGames)
        average = sum(rewards) / len(rewards)
        big_average += average
        if average > max_score:
            max_score = average
    print("Total average with Epsilon = 0: ", str(big_average / 10))
    print("Maximum score with Epsilon = 0: ", max_score)

    """for st in player.qvalues:
        print(st + ": " + str(player.qvalues[st]))"""


# ACTIONS:

actions = []
for a1 in ['keep', 'roll']:
    for a2 in ['keep', 'roll']:
        for a3 in ['keep', 'roll']:
            actions.append(a1 + '-' + a2 + '-' + a3)


# Q LEARNER:

class PlayerQMDP():
    init_state = '9-1-1-1'

    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.results = []
        self.qvalues = {}
        self.exploration_count = 0
        self.exploitation_count = 0
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

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
            q = (1 - self.alpha) * self.qvalues[last][self.action] + self.alpha * (self.reward + self.gamma * self.findMax(state)[0])
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
        print("values : ", self.qvalues[state])
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
