
def at( table, i, default ):
    if( len(table) > i ):
        return table[i]
    return default

class Piece():
    def __init__(self, table ):
        self.position=  int( at( table, 0, 0 ) )
        self.type=      int( at( table, 1, 0 ) )
        self.name=      str( at( table, 2, 0 ) )
        self.owner=     int( at( table, 3, 0 ) )
        self.attributs= []
        if( len(table) > 6 ):
            self.attributs= [ int(x) for x in table[6:] ]

    def asTable(self):
        return [self.type, self.name, self.owner, self.position, len(self.attributs), 'attributes' ] + self.attributs
    
    def copie(self):
        return Piece( self.asTable() )
    
    def __str__(self):
        return f'player-{self.owner}: {self.type} {self.name} on {self.position} {self.attributs}'
