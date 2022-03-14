from . import game

class Interface():
    def __init__(self, sock, player ):
        self.sock= sock
        self.player= player
        self.playerId= -1
        self.score= []
        self.horizon= -1
        self.tabletop= []
        self.pieces= []

    def go(self):
        self.state= self.stWakeUp
        while True :
            try:
                data= self.sock.recv(4096)
            except:
                break
            if not data :
                break
            data= data.decode('utf-8').split('\n')
            for line in data :
                if line != "" :
                    self.state= self.state( line.split(' ') )

    def stWakeUp(self, input):
        if( input[0] == "Info:" ):
            return self.stWakeUp
        elif( input[0] == "Player:" ):
            self.numberOfPlayer= int(input[1])
            self.playerId= int(input[2])
        elif( input[0] == "Tabletop:" ):
            self.tabletop= [ [] for i in range( int(input[1]) ) ]
            self.size= 0
        elif( input[0] == "Node:" ):
            assert( self.size == int(input[1]) )
            for edge in input[4:] :
                self.tabletop[self.size].append( int(edge) )
            self.size+= 1
            if self.size == len(self.tabletop) :
                self.player.wakeUp( self.numberOfPlayer, self.playerId, self.tabletop )
                return self.stLoop
        return self.stWakeUp

    def stLoop(self, input):
        if( input[0] == "Info:" ):
            return self.stLoop
        elif( input[0] == "End:" ):
            self.player.sleep( int(input[1]) )
            return self.stWakeUp
        elif( input[0] == "Player:" ):
            self.playerId= int(input[1])
            self.scores= [ float(val) for val in input[3:] ]
            return self.stLoop
        elif( input[0] == "Game:" ):
            self.turn= int(input[1])
            self.size= int(input[2])
            self.pieces= []
            return self.stLoop
        elif( input[0] == "Piece:" ):
            self.pieces.append( game.Piece( input[1:] ) )
            if self.size == len(self.pieces) :
                self.player.perceive( self.turn, self.scores, self.pieces )
                return self.stLoop
        elif input[0] == "Your-turn:" :
            action= self.player.decide()
            self.sock.send( str.encode(action) )
        return self.stLoop

    def sleep(self):
        print("sleep")
