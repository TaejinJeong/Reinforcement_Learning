import numpy as np
from tqdm import tqdm

class Black_Jack:

    ## Init for the initial state
    def __init__(self, discount_rate = 1, learning_rate = 0.1, exp_rate = 0.3, actions = [0, 1]):

        """
        :param discount_rate: parameter for future return
        :param learning_rate: parameter for learning rate
        :param exp_rate: exploration rate
        :param actions: action space
        """

        # Environment Set-up
        self.lr = learning_rate
        self.exp_rate = exp_rate
        self.discount_rate = discount_rate  # 1 Since all rewards within a game are zero

        # Game Set-up
        self.card_number = list(range(1, 10)) + [10] * 4
        self.actions = actions  # [0, 1] for 0: Stand / 1: Hit
        self.state = (0, 0, False)  # Initial state: (player_hands_sum, dealer_show_card, usable_ace)
        self.stops = False
        self.black_jack = False

        # Player Set-up
        self.player_state_acton = []
        self.player_Q_values = {}  # key: [(player_hands, dealer_show_card, usable_ace)][action] = value
                                   # If the sum is less than 11, then player should hit to get the hands close to 21
        for i in range(12, 22):
            for j in range(1, 11):
                for k in [True, False]:
                    self.player_Q_values[i, j, k] = {}
                    for a in self.actions:
                        if (i == 21) & (a == 0):
                            self.player_Q_values[(i, j, k)][a] = 1
                        else:
                            self.player_Q_values[(i, j, k)][a] = 0
        self.p_black_jack = False

        # Dealer Set-up
        self.dealer_hands = (0, False, False) # inital dealer hands (dealer_hands_sum, usable_ace, busted)
        self.d_black_jack = False

    ## Basic Action
    def Hit(self):
        return np.random.choice(self.card_number)

    def Result(self):
        # Reward => player wins 1 | draw 0 | player loses -1
        reward = 0
        player_sum = self.state[0]
        dealer_sum = self.dealer_hands[0]

        if player_sum > 21:
            if dealer_sum > 21:
                reward = 0
            else:
                reward = -1
        else:
            if dealer_sum > 21:
                reward = 1
            else:
                if player_sum > dealer_sum:
                    reward = 1
                elif player_sum < dealer_sum:
                    reward = -1
                else:
                    reward = 0
        return reward

    def Reset(self):
        self.state = (0, 0, False)
        self.dealer_hands = (0, False, False)
        self.stops = False
        self.player_state_acton = []
        self.black_jack = False
        self.d_black_jack = False
        self.p_black_jack = False

    def Proceed_State(self, action):
        if action == 0:
            self.stops = True
        else:
            player_sum = self.state[0]
            showing_card = self.state[1]
            usable_ace = self.state[2]

            hit = self.Hit()
            if hit == 1:
                usable_ace = True
                player_sum += 11
                if player_sum > 21:
                    player_sum -= 10
            else:
                player_sum += hit

            if player_sum > 21:
                self.stops = True
            self.state = (player_sum, showing_card, usable_ace)

    def Game_Set_up(self):
        ## Distribute cards to dealer / player
        p_hands = (np.random.choice(self.card_number), np.random.choice(self.card_number))
        d_hands = (np.random.choice(self.card_number), np.random.choice(self.card_number))

        # Player Side
        p_usable_ace = False
        for numb in p_hands:
            if numb == 1:
                p_usable_ace = True
                if sum(p_hands) == 11:
                    self.p_black_jack = True
        if p_usable_ace:
            p_hands = sum(p_hands) + 10
        else:
            p_hands = sum(p_hands)
        self.state = (p_hands, d_hands[0], p_usable_ace)

        # Dealer Side
        d_usable_ace = False
        busted = False
        for numb in d_hands:
            if numb == 1:
                d_usable_ace = True
                if sum(d_hands) == 11:
                    self.d_black_jack = True
        if d_usable_ace:
            d_hands = sum(d_hands) + 10
        else:
            d_hands = sum(d_hands)
        self.dealer_hands = (d_hands, d_usable_ace, busted)

    def Check_Black_Jack(self):
        if self.p_black_jack:
            if self.d_black_jack:
                reward = 0
            else:
                reward = 1
            self.black_jack = True
        else:
            if self.d_black_jack:
                reward = -1
                self.black_jack = True
            else:
                reward = 0
        return reward

    def Reward_Update(self):
        reward = self.Result()
        # backpropagate reward
        for s in reversed(self.player_state_acton):
            state, action = s[0], s[1]
            reward = self.player_Q_values[state][action] + self.lr*(reward - self.player_Q_values[state][action])
            self.player_Q_values[state][action] = round(reward, 4)

    ## Dealer's part
    def Dealer_Policy(self):
        dealer_hands, usable_ace, busted = self.dealer_hands
        if dealer_hands >= 17:
            return
        else:
            while dealer_hands < 17:
                card = self.Hit()
                if card == 1 and not usable_ace:
                    # whether or not dealer has an ace in their hand, only one ace can be switched to 11
                    # (If the dealer uses the two aces to 11 each, then the dealer must be busted)
                    usable_ace = True
                    dealer_hands += card + 10
                    if dealer_hands > 21:
                        dealer_hands -= 10
                else:
                    dealer_hands += card
            if dealer_hands > 21:
                busted = True
            self.dealer_hands = (dealer_hands, usable_ace, busted)

    ## Player's Part
    def Action(self):
        # If player_sum is less than 12, then player should hit
        player_sum = self.state[0]
        if player_sum <= 11:
            return 1
        else:
            # greedy action
            action = self.actions[0]
            state_value = self.player_Q_values[self.state][action]
            for a in self.actions[1:]:
                if self.player_Q_values[self.state][a] > state_value:
                    state_value = self.player_Q_values[self.state][a]
                    action = a

            # Epsilon Greedy Methods
            if np.random.uniform(0, 1) < self.exp_rate:
                ind = self.actions.index(action)
                new_a = self.actions[:ind] + self.actions[ind + 1:]
                action = np.random.choice(new_a)

        return action

    ## Play Black Jack
    def Play(self, rounds = 100000):
        for i in tqdm(range(rounds)):

            ## Inital Set up for the game (Distribute two cards to the dealer and the player)
            self.Game_Set_up()

            ## Plain Game
            if not self.black_jack:
                ## Player Action
                while not self.stops:
                    action = self.Action()
                    if self.state[0] >= 12:
                        state_action = [self.state, action]
                        self.player_state_acton.append(state_action)
                    self.Proceed_State(action)

                ## Dealer Action
                self.Dealer_Policy()

                ## Update reward
                self.Reward_Update()

            ## Reset the game
            self.Reset()