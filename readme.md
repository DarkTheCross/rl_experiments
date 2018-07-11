# This is my experiments on deep reinforcement learning

Several environments from the [OpenAI Gym](https://gym.openai.com/envs/CartPole-v0/) are solved using varieties of reinforcement learning methods.

I'm using python 3.5, tensorflow-gpu. OpenAI gym is also required to reproduce the result.

In cartpoleSolver_.py, the Q-network takes observation and action as input, and the output is the Q value for the state-action pair.

In cartpoleSolver.py, the Q-network takes observation as the input, while the outputs stands for the Q value for the two actions at this state.

cartpoleSolver_.py is more stable because of better hyper parameters.

![sample](img/456.png)


Besides, I also implemented an Apple Picker game (Simplified) and used deep Q learning to implement an AI for the game. You
 can see the performance of the AI during the first 300 iterations and last 300 iterations below.

![first 300](img/first300.gif)

First 300 iterations

![last 300](img/last300.gif)

Last 300 iterations (Solved!)

As for the atari game, I'm using the template from Berkely CS294 course for deep q learning with double q networks.

The algorithm achieved a average reward of over 20 on Pong-v0 after 4.3M of game steps.

![Pong]( img/pong.gif )

GGWP.
