import numpy as np
import torch
from torch.distributions import Normal
from rules import PokerRank
from mapper import Mapper
from typing import Optional


class PokerSimulator:
    """ 
    ### A Terminal Texas Hold'em Poker Simulator. 
    The player is always player 0
    and the other players are bots. The bots operate on simple 
    default probabilities for their actions. The player can choose
    between fold, check, raise, and all-in. The bots will mostly
    check and fold, and rarely raise or go all-in from stronger hands.
    Playing strategic long-term one should be able to win against the bots.

    Args:
        n_players (int): number of players in the game
        buy_in (int): buy-in of the game
        small_blind (int): small blind of the game

    """
    def __init__(self, 
                 n_players : int = 6, 
                 buy_in : int = 1000, 
                 small_blind : int = 10,
                 big_blind : int = 20):
        
        self.n_players = n_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.last_bet = big_blind
        self.mapper = Mapper()
     
        self.n_rounds = 0
        self.player_money = {player: buy_in for player in range(self.n_players)}
        self.order_of_play = [player for player in range(self.n_players)]
        # initialize the game
        self.instantiate()
    
    def instantiate(self):
        """ 
        Reset for next round
        """
        print('\nStarting new round')
        if self.n_rounds > 0:
            self.rotate()
        
        self.last_bet = self.big_blind
        players = [player for player in self.player_money.keys() if self.player_money[player] > 0]
        self.n_players = len(players)
        self.decision_holder = {player : None for player in players}
        self.player_bets = {player : 0 for player in players}
        deck = np.arange(52)
        self.cards_to_table = np.random.choice(deck, 5 + self.n_players * 2, replace=False)
        self.deck = np.setdiff1d(deck, self.cards_to_table)
        cards_on_table = self.cards_to_table[:5]

        self.cards_on_table = [self.get_card(card) for card in cards_on_table]
        self.player_cards = self.cards_to_table[5:].reshape(self.n_players, 2)
        self.player_hands = {}
        
        for i, player in enumerate(players):
            cards = self.player_cards[i]
            self.player_hands[player] = [self.get_card(card) for card in cards]
            if player == 0:
                print('\nYour cards:')
                self.print_cards(self.player_hands[player])
                
        self.n_rounds += 1
        
    
    def get_suit(self, card : int):
        """ 
        Get the suit of a card

        Args:
            card (int): a card

        Returns:
            suit (int): the suit of the card
        """
        return card // 13
    
    def get_rank(self, card : int):
        """ 
        Get the rank of a card

        Args:
            card (int): a card

        Returns:
            rank (int): the rank of the card
        """
        return card % 13
    
    def get_card(self, card : int):
        """ 
        1-4: 2 of clubs, 2 of diamonds, 2 of hearts, 2 of spades
        5-8: 3 of clubs, 3 of diamonds, 3 of hearts, 3 of spades etc.

        Args:
            card (int): a card
        
        Returns:
            card as tuple (suit, rank)
        """
        suit = self.get_suit(card)
        rank = self.get_rank(card)
        return (suit, rank)

    def set_bot_probs(self, hand_strength : int, cards_shown : int):
        """ 
        Set the probabilities for the bot player based on hand strength.
        This is just some dummy probabilities to make the bot play and to 
        train an RL agent.

        Args:
            hand_strength (int): the strength of the hand
            cards_shown (int): number of cards shown on the table
        
        Returns:
            probs (list): list of probabilities for each action
        """ 
        if hand_strength <= 6 and cards_shown == 0:
            probs = [0.2, 0.7, 0.09, 0.01]
        elif hand_strength > 6 and cards_shown == 0:
            probs = [0.08, 0.8, 0.1, 0.02]
        elif hand_strength <= 13 and cards_shown == 3:
            probs = [0.3, 0.6, 0.09, 0.01]
        elif hand_strength > 13 and cards_shown == 3:
            probs = [0.05, 0.75, 0.19, 0.01]
        elif hand_strength <= 13 and cards_shown == 4:
            probs = [0.8, 0.18, 0.01, 0.01]
        elif hand_strength > 13 and cards_shown == 4:
            probs = [0.1, 0.8, 0.09, 0.01]
        elif hand_strength <= 13 and cards_shown == 5:
            probs = [0.9, 0.08, 0.01, 0.01]
        elif hand_strength > 13 and cards_shown == 5:
            probs = [0.1, 0.8, 0.09, 0.01]
      
        
        action = np.random.choice(['fold', 'check', 'raise', 'all-in'], p=probs)
        return action
    
    def verify_bet(self, player : int, amount : float):
        """
        Check if the player can:
        1. afford the bet
        2. has enough money to match the bet
        
        Update the last bet, player bets, player money, and is all-in

        Returns:
            valid (bool): True if the bet is valid, False otherwise
        """
        if amount > self.player_money[player]:
            amount = self.player_money[player]
            self.player_money[player] = 0
            valid = False
        
        else:
            self.player_money[player] -= amount
            valid = True
        
        self.player_bets[player] += amount
        self.last_bet = self.player_bets[player] if self.player_bets[player] > self.last_bet else self.last_bet

        return valid
    
    def remove_player(self, player : int):
        """ 
        Remove player from round
        """
        self.n_players -= 1
        del self.player_hands[player]
        del self.decision_holder[player]
        del self.player_bets[player]
        del self.player_money[player]
        self.order_of_play.remove(player)
        
    def fold(self, player : int):
        """ 
        If fold is called, the player is out of the game.
        """
        # check if player is either small blind or big blind
        if self.order_of_play[player] == self.n_players - 2 and self.player_bets[player] < self.small_blind:
            amount = self.small_blind - self.player_bets[player]
            self.verify_bet(player=player, amount=amount)
        elif self.order_of_play[player] == self.n_players - 1 and self.player_bets[player] < self.big_blind:
            amount = self.big_blind - self.player_bets[player]
            self.verify_bet(player=player, amount=amount)
        self.n_players -= 1
        print(f'Player {player+1} folds | Amount left {self.player_money[player]:.0f}')
        print('-'*50)
    
    def check(self, player : int):
        """ 
        Check and call is used interchangeably.
        If check is called, the player can continue to play.
        """
        if self.player_bets[player] <= self.last_bet:
            amount = self.last_bet - self.player_bets[player]
            # check if player can afford the bet
            self.last_bet = self.player_bets[player] if self.player_bets[player] > self.last_bet else self.last_bet
            valid = self.verify_bet(player=player, amount=amount)
            decision = 'checks/calls' if valid else 'all-in'

            print(f'Player {player+1} {decision} | Amount left: {self.player_money[player]:.0f}')
            print('-'*50)
            
    def raise_money(self, player : int,):
        """ 
        The raise is drawn from a normal distribution with mean 1/5 of the player's money
        and standard deviation 1/10 of the player's money. Minimum raise is the big blind.
        """
        player_money = self.player_money[player]
        if player_money < self.big_blind:
            self.all_in(player=player)
        else:
            bet_dist = Normal(loc=player_money / 5, scale=player_money / 10)
            amount = torch.clamp(bet_dist.sample(), min=self.last_bet, max=player_money).item()
            valid = self.verify_bet(player=player, amount=amount)
            decision = 'raises' if valid else 'all-in'
            print(f'Player {player+1} {decision} {amount:.0f} | Amount left: {self.player_money[player]:.0f}')
            print('-'*50)
        

    def all_in(self, player : int):
        """ 
        The all-in matches the money of the player calling the all-in.
        """
        # check that player has enough money
        amount = self.player_money[player]
        if amount < self.big_blind:
            amount = self.big_blind

        self.verify_bet(player=player, amount=amount)
        print(f'Player {player+1} all-in {amount:.0f} | Amount left: {self.player_money[player]:.0f}')
        print('-'*50)

    def rotate(self):
        """
        move all values one step forward, last value becomes first
        """
        order_copy = self.order_of_play.copy()
        self.order_of_play[0] = order_copy[-1]
        self.order_of_play[1:] = order_copy[:-1]
    
    def player_input(self):
        """ 
        Player action
        """
        action = input('(f)old, (c)heck, (r)aise, (a)ll-in): ')
        print('\n')
        if action == 'f':
            return 'fold'
        elif action == 'c':
            return 'check'
        elif action == 'r':
            return 'raise'
        elif action == 'a':
            return 'all-in'
        else:
            raise ValueError('Invalid action')
    
    def print_winner(self, winner : list):
        """ 
        Determine single winner before river
        """
        winner_list = [f'Player {w+1}' for w in winner]
        winner = ', '.join(winner_list)
        print(f'\nWinner(s) is/are: {winner}\n')
    
    def distribute_pot(self, player : list):
        """
        Distribute the pot to the winner(s)
        """
        self.print_winner(winner=player)
        # TODO use player bets to distribute pot
        # if any player is all-in and wins, we must re-distribute if the player's all-in is 
        # less than the last bet. Players above the all-in will get their money back.
        # if any player money is at zero and did not win, then the player is out of the game
        pass 

    def is_game_over(self):
        """ 
        Check if only one player still has money to play
        """
        money = list(self.player_money.values())
        # if only one player has money left, the player wins and the game is over
        count = [m >= self.big_blind for m in money]
        if sum(count) == 1:
            return True
        return False
    
    def showdown(self, players : list):
        """ 
        Showdown
        """
        player_hands = {player : self.player_hands[player] for player in players}
        print('\n')
        for player, cards in player_hands.items():
            print(f'Player {player+1}: {self.mapper.card_string(cards[0])} {self.mapper.card_string(cards[1])}')
            print('-'*50)
        rules = PokerRank(player_hands=player_hands,
                          table_cards=self.cards_on_table,
                          verbose=True)
        winner = rules.get_winner()
        self.distribute_pot(winner)
    
    def print_cards(self, cards):
        """ 
        Print cards
        """
        cards_string = [f'|{self.mapper.card_string(card)}|' for card in cards]
        cards = ' '.join(cards_string)
        print('-'*50)
        print(f'{cards}')
        print('-'*50 + '\n')

    def player_action(self, player : int, action : str):
        """ 
        Action of a player
        """
        if action == 'fold':
            self.fold(player=player)
        elif action == 'check':
            self.check(player=player)
        elif action == 'raise':
            self.raise_money(player=player)
        elif action == 'all-in':
            self.all_in(player=player)
        else:
            raise ValueError('Invalid action')
        
        self.decision_holder[player] = action
    
    def get_players_in_round(self, player : int, action : str):
        """ 
        Update which players are still in the round, and the decision(s)
        based on subsequent player actions
        """
        if action in ['raise', 'all-in']:
            # update the decision holder for all players that have not made a decision
            idx = self.order_of_play.index(player)
            # set to none for all players before the player that raised or went all-in
            for i in range(idx):
                if self.decision_holder[self.order_of_play[i]] not in ['fold', 'all-in']:
                    self.decision_holder[self.order_of_play[i]] = None
        
    def player_moves(self, table_cards : list = None, cards_shown : int = 0):
        """ 
        Round of betting
        """
        players = list(self.decision_holder.keys())
        bot_hands = {player: self.player_hands[player] for player in self.decision_holder.keys() if player != 0}
        
        ranks = PokerRank(
            player_hands=bot_hands,
            table_cards=table_cards,
            verbose=False
        )
        # used for bot decision probabilities
        bot_ranks = ranks.rank_player_hands
        n_players = self.n_players
        while True:
            for player in self.order_of_play:
                if all([self.player_money[player] == 0 for player in players]):
                    print(f'Player(s) money, {self.player_money}. Round over')
                    return True
                elif all([self.decision_holder[player] in ['fold', 'all-in'] for player in self.decision_holder.keys()]):
                    print(f'Decisions: {self.decision_holder}. Round over')
                    return True
                elif all([decision is not None for decision in self.decision_holder.values()]):
                    print(f'Decisions: {self.decision_holder}. Round not over')
                    return False    
                elif self.player_money[player] == 0:
                    self.decision_holder[player] = 'all-in'
                    print(f'Decisions: {self.decision_holder}')
                    continue
                elif self.decision_holder[player] in ['fold', 'all-in']:
                    continue
                
                if player == 0 and self.decision_holder[player] is None:
                    action = self.player_input()
                    self.player_action(player=player, action=action)
                    self.get_players_in_round(player, action)
                
                elif self.decision_holder[player] is None:
                    hand_strength = bot_ranks[player]
                    action = self.set_bot_probs(hand_strength=hand_strength, cards_shown=cards_shown)
                    self.player_action(player=player, action=action)
                    self.get_players_in_round(player,action)
                
                if action == 'fold':
                    n_players -= 1
                    if n_players == 1:
                        return True


    def poker_round(self):
        """ 
        Simulate a round of poker
        (Pre-flop, Flop, Turn, River)
        """
        stages = ['Pre-flop', 'Flop', 'Turn', 'River']
        cards_shown = [0, 3, 4, 5]

        def players_in():
            players = []
            for player in self.decision_holder.keys():
                if self.decision_holder[player] != 'fold':
                    players.append(player)
            return players
        
        for cards, stage in zip(cards_shown, stages): 
            table_cards = self.cards_on_table[:cards]
            print(f'\n{stage}')
            if stage != 'Pre-flop':
                self.print_cards(table_cards)
        
            round_over = self.player_moves(table_cards=table_cards, cards_shown=cards)
            
            if round_over or stage == 'River':
                self.showdown(players=[player for player in self.decision_holder.keys() if self.decision_holder[player] != 'fold'])
                break
            else:
                players = players_in()
                self.decision_holder = {player : None if player in players else self.decision_holder[player] for player in self.decision_holder.keys()}
    
    def play(self):
        """ 
        Play rounds of poker until the player decides to stop
        or only one player is left.
        """
        ctn = True
        self.poker_round()
        # TODO remove players with no money left
        # TODO distribution of pot
        while ctn:
            if self.is_game_over():
                print('Game over')
                break
            ctn = input('Continue? (y/n): ')
            if ctn == 'y':
                ctn = True
                self.instantiate()
                self.poker_round()
            else:
                ctn = False

if __name__ == "__main__":
    N_PLAYERS = 6 # number of bots + 1 player
    BUY_IN = 1e4
    SMALL_BLIND = BUY_IN / 100
    BIG_BLIND = SMALL_BLIND * 2
    
    game = PokerSimulator(
        n_players=N_PLAYERS,
        buy_in=BUY_IN,
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND
    )
    game.play()
