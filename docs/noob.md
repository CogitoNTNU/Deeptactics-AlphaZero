
# Temperatur når man skal velge trekk

[10, 5, 15, 10, 20, 0, 10, 10, 10] - antall visits

listen = [10, 5, 15, 10, 20, 0, 10, 10, 10] ** 1/t - legg til temperatur på antall visits 

softmax(listen) - gjør at det summer til 1

np.random.choiche(listen, probability = listen)

(i alpha GO, etter 30 trekk, endre random.choice(listen) -> max(listen))

# Dirichlet støy i sannsynlighetene fra

Bare i rotnoden

bruker numpy.random.dirichlet(value_probability)

1. copy alphhazero - DONE
2. implement dirichlet noise - DONE
3. implement temperature for move selection before x moves. - DONE
4. implement max value selection etter x trekk, før np.random.choice - DONE
5. Lage self play games. Alphazero vs. Alphazero 
6. Regne ut entropy_loss fra policy network, MSE Loss på value network, og L2 regulization 
7. backpropagation basert på self play games

## Definition of the PUCT formula

PUCT:

```python
 def PUCT(self, node: Node) -> float:
        if node.visits == 0:
            Q = 0  # You don't know the value of a state you haven't visited. Get devision error
        else:
            Q = node.value / node.visits  # Take the average

        U = (self.c * node.policy_value * np.sqrt(node.parent.visits) / (1 + node.visits))

        PUCT = Q + U

        return PUCT
```

node.visits = [3, 4, 5]
node.value = [7 , 2, 10]
node.policy_values = [0.1, 0.6, 0.3]
c = [1.41, 1.41, 1.41]
sqrt_parent_visits = [sqrt(12), sqrt(12), sqrt(12)]


Save list of moves made during the game.
[(state, normalized_number_of_visits, actual_winner)]



Actual winner is 1 if player 1 wins, -1 if player 2 wins, 0 if draw

(state) -> (probabilities, value)
