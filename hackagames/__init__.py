#!env python3
import socket
from . import abstract, client

def takeASeat(host, port, player ):
    with socket.socket( socket.AF_INET, socket.SOCK_STREAM) as sock :
        sock.connect((host, port))
        itf= client.Interface( sock, player )
        itf.go()

class Player(abstract.Player) :
    def __init__(self):
        self.results= []

    def wakeUp(self, numberOfPlayers, playerId, tabletop):
        self.turn= 0
        self.id= 0
        self.pieces= []
        self.scores= [0]
        self.numberOfPlayers= numberOfPlayers
        self.id= playerId
        self.tabletop= tabletop
        
    def perceive(self, turn, scores, pieces, deltaTabletop):
        self.turn= turn
        self.pieces= pieces
        self.scores= scores

    def decide(self):
        return "sleep"

    def sleep(self, result):
        self.results.append(result)

class PlayerVerbose(Player):
    def wakeUp(self, numberOfPlayers, playerId, tabletop):
        super().wakeUp(numberOfPlayers, playerId, tabletop)
        print( "Tabletop:"),
        for i in range( len(self.tabletop) ):
            print( '  ', str(i), ':\t', str(self.tabletop[i]) )
    
    def perceive(self, turn, scores, pieces, deltaTabletop=[]):
        super().perceive(turn, scores, pieces, deltaTabletop)
        print( f'player-{self.id}" turn: {turn}' )
        print( 'Pieces:', ',\n\t'.join([ str(p) for p in self.pieces ]) )
        print( 'score:', scores)

    def decide(self):
        a= super().decide()
        print( 'action:', a)
        return a

    def sleep(self, result):
        super().sleep(result)
        print( "Final: ", result)
