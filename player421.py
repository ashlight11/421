#!env python3
"""
Script MDP 421 
"""
import random, hackagames as hg

actions = []
for a1 in ['keep', 'roll']:
    for a2 in ['keep', 'roll']:
        for a3 in ['keep', 'roll']:
            actions.append(a1 + '-' + a2 + '-' + a3)


class PlayerRandom(hg.abstract.Player):
    def __init__(self):
        self.results = []

    # AI interface :
    def wakeUp(self, numberOfPlayers, playerId, tabletop):
        self.scores = [0 for i in range(numberOfPlayers)]
        self.id = playerId
        self.model = tabletop

    def perceive(self, turn, scores, pieces):
        self.reward = scores[self.id] - self.scores[self.id]
        self.scores = scores
        self.turn = turn
        self.dices = pieces

    def decide(self):
        action = random.choice(actions)
        return action

    def sleep(self, result):
        self.results.append(result)


class PlayerHuman(PlayerRandom):

    def wakeUp(self, numberOfPlayers, playerId, tabletop):
        super().wakeUp(numberOfPlayers, playerId, tabletop)
        print('New Game...')

    def perceive(self, turn, scores, pieces):
        super().perceive(turn, scores, pieces)
        print(f'Dice: {str(self.dices)} Horizon: {self.turn} Reward: {self.reward}')

    def decide(self):
        print("Action ?")
        action = ""
        while not action in actions:
            action = input()
        return action
