

class Player() :
    # AI interface :
    def wakeUp(self, numberOfPlayers, playerId, tabletop):
        pass

    def perceive(self, turn, scores, pieces, deltaTabletop):
        pass

    def decide(self):
        pass # must return the action to perfom as a string.

    def sleep(self, result):
        pass

class Engine() :
    # Game engine interface :
    def initialize(self):
        pass
    
    def turn():
        pass

    def pieces():
        pass
    
    def step(self, playerAction):
        pass

    def step(self):
        pass

    def start(self, players, numberOfGames=1):
        pass

