# 421, an HackaGames game (hum)

**421** is simple dice games where one unique player try to optimize it dice combination after a maximum of 2 roll dice steps.

## Installation

**421** (as **HackaGames**) is natively developed on and for Linux systems.
Commands are given regarding Ubuntu-like distribution.

Download and unzip the game archive (on Linux machines):

- [hackagames-421.zip](https://bitbucket.org/imt-mobisyst/hackagames/raw/master/release/hackagames-421.zip)


## Try the game:

**421** is not a standard **HackaGames** game, and it works locally with a unique player implemented in Python regarding **HackaGames** **Player** interface.

The `play.py` script launch a **421** game with a human player on the terminal.

In a terminal:

```sh
./play.py
```

The player can perform one and only one action at it turns, and the game stops automatically after 2 turns.

The actions consist in keeping or rolling each of the 3 dices. So there are 8 actions:

- `keep-keep-keep`,  `keep-keep-roll`,  `keep-roll-keep`,  `keep-roll-roll`, `roll-keep-keep`,  `roll-keep-roll`,  `roll-roll-keep` and `roll-roll-roll`

The goal is to optimize the combination of dices before the end of the 2 turns. The best combination ever is **421**. But you can explore other combinations.

## Your first AI:

The file `player421.py` propose a first random AI with the required structure to play **421**.
So copy this player and start to implement simple ideas...

```bash
cp player421.py myBeatifullAI.py
```

You can try your *AI* by modifying the `play.py` scripts.

