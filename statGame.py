#!env python3
from game421 import Engine

print("\nSimple 421 implementation")

game= Engine()

print("\nStates: ")

for state in game.allStates() :
    game.setOnStateDico( state )
    print( game.stateStr() + ": is end ? "+ str(game.isEnd()) + " ; value "+ str(game.score( game.state )) )

print( str(len( game.allStates() )) + " states" )

print("\nActions: ")

print( ", ".join([ game.actionToStr(a) for a in game.allActions() ]) )

print( str(len( game.allActions() )) + " actions" )
