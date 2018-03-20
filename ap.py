import numpy as np
import random

class ApplePicker:
    def __init__(self):
        self.gameState = {"board": np.random.choice(5, 1), "y": 0, "x": np.random.choice(7, 1)}

    def render(self):
        gameScene = np.zeros((7,7), dtype=np.float32)
        gameScene[6, int(self.gameState["board"]):int(self.gameState["board"])+1] = 1
        gameScene[self.gameState["y"], self.gameState["x"]] = 1
        return gameScene

    def reset(self):
        self.gameState["y"] = 0
        self.gameState["x"] = random.randint(0, 6)
        return self.render()

    def step(self, act):
        rwd = 0
        done = False
        self.gameState["board"] += act-1
        self.gameState["board"] = max(self.gameState["board"], 0)
        self.gameState["board"] = min(self.gameState["board"], 6)
        self.gameState["y"] += 1
        if self.gameState["y"] == 6:
            if self.gameState["x"] >= self.gameState["board"] and self.gameState["x"] < self.gameState["board"] + 1:
                rwd = 1
            else:
                rwd = -1
            done = True
        obs = self.render()
        return obs, rwd, done
