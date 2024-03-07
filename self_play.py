"""
game = []

...
...
...
value = 1
predicted value = -0.4
Value loss = Mean squared error mellom value og predicted value

actions fra mcts med nevralnett = [0.1, 0.1, 0.1, 0.6, 0.2, 0.1]
predicted actions = [0.5, 0.2, 0.1, ...]
Action loss = Cross enthropy loss mellom predicted actions og actions fra mcts med nevralnett

...
.x.
...
value = -1
predicted value = 0.2


o..
.x.
...
value = 1

0.x
.x.
...
value = -1


osv.

0.x
0x.
x..

"""