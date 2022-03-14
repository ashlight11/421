#!env python3
import game421 as game
import player421 as player

gameEngine = game.Engine()
player = player.PlayerHuman()

numberOfGames = 2
rewards = gameEngine.start(player, numberOfGames)
print(rewards)
