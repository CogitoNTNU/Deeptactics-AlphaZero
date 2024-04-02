
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
2. implement dirichlet noise
3. implement temperature for move selection before x moves.
4. implement max value selection etter x trekk, før np.random.choice 
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
