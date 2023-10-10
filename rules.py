import numpy as np
from mapper import Mapper

class PokerRank:
    """ 
    A Texas Hold'em Poker Rules
    where we have 52 cards, 4 suits, and 13 ranks

    Cards are represented by integers from 0 to 51
    where the suit is the integer division of the card by 13
    and the rank is the remainder of the division of the card by 13.

    The hand is combined with the table cards where each card is a tuple (suit, rank).
    """

    def __init__(self, player_hands : dict, table_cards : list, verbose : bool = False):
        if table_cards is not None:
            self.player_hands = {player : player_hands[player] + table_cards for player in player_hands}
        else:
            self.player_hands = player_hands
        
        self.mapper = Mapper()       
        self.rank_player_hands = {}
        for player in self.player_hands:
            self.rank_player_hands[player] = self.rank_hand_of_player(player, verbose)
    
    def is_royal_flush(self, hand : list):
        """ 
        Check if a hand is a royal flush

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            is_royal_flush (bool): True if the hand is a royal flush, False otherwise
        """
        flush, flush_cards = self.is_flush(hand)
        straight = False

        if flush:
            straight, straight_cards = self.is_straight(flush_cards)
        
        if (flush & straight) and (self.get_rank_of_highest_card(hand) == 12):
            return True, straight_cards
        
        return False, None
    
    def is_straight_flush(self, hand : list):
        """ 
        Check if a hand is a straight flush. 

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            is_straight_flush (bool): True if the hand is a straight flush, False otherwise
        """
        flush, flush_cards = self.is_flush(hand) 
        straight = False
        
        if flush:
            straight, straight_cards = self.is_straight(flush_cards)
        
        if flush and straight:
            return True, straight_cards
        
        return False, None
    
    def is_flush(self, cards : list):
        """ 
        Check if a hand is a flush. 

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            is_flush (bool): True if the hand is a flush, False otherwise
        """
        if len(cards) < 5:
            return False, None
        
        suits = [card[0] for card in cards]
        # check if we have 5 cards of the same suit
        if np.any(np.bincount(suits) >= 5):
            # get the cards of the flush
            flush_cards = [card for card in cards if card[0] == np.argmax(np.bincount(suits))]
            # get the highest 5 cards of the flush
            flush_cards.sort(key=lambda card: card[1], reverse=True)
            flush_cards = flush_cards[:5]
            
            return True, flush_cards
        
        return False, None
    
    def is_straight(self, cards : list):
        """ 
        Check if a hand is a straight. 

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            is_straight (bool): True if the hand is a straight, False otherwise
        """
        ranks = [card[1] for card in cards]
        ranks.sort()
        # remove any duplicates
        ranks = np.unique(ranks)
        if len(ranks) < 5:
            return False, None
        
        cards = [card for card in cards if card[1] in ranks]
        # remove duplicate ranks from cards

        if len(ranks) < 5:
            return False, None
        
        def unique_ranks(cards):
            """return only 5 highest ranks with no duplicates"""
            ranks, _cards = [], []
            for card in cards:
                if card[1] not in ranks:
                    ranks.append(card[1])
                    _cards.append(card)
                else:
                    continue
            _cards.sort(key=lambda card: card[1], reverse=True)
            return _cards[:5]

        # take highest straight first
        if np.all(np.diff(ranks[2:7]) == 1) and len(ranks[2:7]) == 5:
            straight_cards = [card for card in cards if card[1] in ranks[2:7]]
            straight_cards = unique_ranks(straight_cards)
            return True, straight_cards
        
        if np.all(np.diff(ranks[1:6]) == 1) and len(ranks[1:6]) == 5:
            straight_cards = [card for card in cards if card[1] in ranks[1:6]]
            straight_cards = unique_ranks(straight_cards)
            return True, straight_cards
        
        if np.all(np.diff(ranks[:5]) == 1) and len(ranks[:5]) == 5:
            straight_cards = [card for card in cards if card[1] in ranks[:5]]
            straight_cards = unique_ranks(straight_cards)
            return True, straight_cards
        
        return False, None

    
    def is_4_of_a_kind(self, cards : list):
        """ 
        Check if a hand is a 4 of a kind. 

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            is_4_of_a_kind (bool): True if the hand is a 4 of a kind, False otherwise
        """
        ranks = np.array([card[1] for card in cards])
        # get the unique ranks and their counts
        _, counts = np.unique(ranks, return_counts=True)
        
        if np.any(counts == 4):
            # get the cards of the 4 of a kind
            four_of_a_kind_cards = [card for card in cards if card[1] == np.argmax(np.bincount(ranks))]
            cards = [card for card in cards if card not in four_of_a_kind_cards]
            # get the highest card of the remaining cards
            highest_card = self.get_highest_card(cards)
            hand = four_of_a_kind_cards + [highest_card]
            return True, hand
        
        return False, None
    
    def is_full_house(self, cards : list):
        """ 
        Check if a hand is a full house (3 of a kind and 1 pair).

        Args:
            hand (list): a list of 2 tuples (cards)
        
        Returns:
            is_full_house (bool): True if the hand is a full house, False otherwise
        """
        ranks = np.array([card[1] for card in cards])
        _, counts = np.unique(ranks, return_counts=True)
        
        # TODO could be two 3 of a kind and two pairs, need to check for that

        if np.any(counts == 3) and np.any(counts == 2):
            three_of_a_kind_cards = [card for card in cards if card[1] == np.argmax(np.bincount(ranks))]
            cards = [card for card in cards if card not in three_of_a_kind_cards]
            ranks = np.array([card[1] for card in cards])
            two_of_a_kind_cards = [card for card in cards if card[1] == np.argmax(np.bincount(ranks))]
            hand = three_of_a_kind_cards + two_of_a_kind_cards
            return True, hand

        return False, None
    
    def is_3_of_a_kind(self, cards : list):
        """ 
        Check if a hand is a 3 of a kind.

        Args:
            hand (list): a list of 2 tuples (cards)
        
        Returns:
            is_3_of_a_kind (bool): True if the hand is a 3 of a kind, False otherwise
        """
        ranks = np.array([card[1] for card in cards])
        _, counts = np.unique(ranks, return_counts=True)
        
        if np.any(counts == 3):
            three_of_a_kind_cards = [card for card in cards if card[1] == np.argmax(np.bincount(ranks))]
            cards = [card for card in cards if card not in three_of_a_kind_cards]
            cards.sort(key=lambda card: card[1], reverse=True)
            
            for i in range(2):
                highest_card = self.get_highest_card(cards[i:])
                three_of_a_kind_cards.append(highest_card)

            hand = three_of_a_kind_cards
            return True, hand
        
        return False, None
    
    def is_2_pairs(self, cards : list):
        """ 
        Check if a hand is a 2 pairs.

        Args:
            hand (list): a list of 2 tuples (cards)
        
        Returns:
            is_2_pairs (bool): True if the hand is a 2 pairs, False otherwise
        """
        ranks = np.array([card[1] for card in cards])

        arr, counts = np.unique(ranks, return_counts=True)
        
        if np.sum(counts == 2) == 2:
            # find the 2 pairs
            pairs = arr[np.where(counts == 2)]
            hand = [card for card in cards if card[1] in pairs]
            cards = [card for card in cards if card not in hand]
            highest_card = self.get_highest_card(cards)
            hand.append(highest_card)
            return True, hand
        
        return False, None
    
    def is_1_pair(self, cards : list):
        """ 
        Check if a hand is a 1 pair.

        Args:
            hand (list): a list of 2 tuples (cards)
        
        Returns:
            is_1_pair (bool): True if the hand is a 1 pair, False otherwise
        """
        ranks = np.array([card[1] for card in cards])

        arr, counts = np.unique(ranks, return_counts=True)

        if np.any(counts == 2):
            # get the rank that appears 2 times
            pair = arr[np.where(counts == 2)][0]
            hand = [card for card in cards if card[1] == pair]
            cards = [card for card in cards if card not in hand]
            cards.sort(key=lambda card: card[1], reverse=True)
            if len(cards) < 3:
                return True, hand
            for i in range(3):
                highest_card = self.get_highest_card(cards[i:])
                hand.append(highest_card)
            return True, hand
        
        return False, None
    
    def is_high_card(self, cards : list):
        """ 
        Check if a hand is a high card.

        Args:
            hand (list): a list of 2 tuples (cards)
        
        Returns:
            is_high_card (bool): True if the hand is a high card, False otherwise
        """
        # get the 5 highest cards
        cards.sort(key=lambda card: card[1], reverse=True)
        hand = cards[:5]
        return True, hand
    
    def get_highest_card(self, cards):
        """ 
        Get the highest card of a hand.
        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            card (tuple): the highest card
        """
        cards_str = [self.mapper.card_string(card) for card in cards]

        cards.sort(key=lambda card: card[1], reverse=True)
        return cards[0]
    
    
    def get_rank_of_highest_card(self, cards : list):
        """ 
        Get the rank of the highest card of a hand.
        (exludes the cards needed for making a pair, 3 of a kind, etc.)
        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            rank (int): the rank of the highest card
        """
        ranks = [card[1] for card in cards]
        ranks.sort()
        return ranks[-1]
    
    # all functions below get a hand of five cards (2 from player and 3 from table)
    # the hand is a list of tuples (suit, rank)
    
    def get_rank_of_4_of_a_kind(self, hand : list):
        """ 
        Get the rank of the 4 of a kind of a hand

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            rank (int): the rank of the 4 of a kind
        """
        # get 4 of a kind rank
        ranks = [card[1] for card in hand]
        arr, counts = np.unique(ranks, return_counts=True)
        rank = arr[np.where(counts == 4)][0]
        
        return 91 + rank
    
    def get_rank_of_3_of_a_kind(self, hand : list):
        """ 
        Get the rank of the 3 of a kind of a hand

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            rank (int): the rank of the 3 of a kind
        """
        # get 3 of a kind rank
        ranks = [card[1] for card in hand]
        arr, counts = np.unique(ranks, return_counts=True)
        rank = arr[np.where(counts == 3)][0]
        
        return 39 + rank
    
    def get_rank_of_2_pairs(self, hand : list):
        """ 
        Get the rank of the 2 pairs of a hand

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            rank (int): the rank of the 2 pairs
        """
        # get 2 pairs ranks
        ranks = [card[1] for card in hand]
        arr, counts = np.unique(ranks, return_counts=True)
        pairs = arr[np.where(counts == 2)]
        rank = np.max(pairs)
        
        return 26 + rank
    
    def get_rank_of_1_pair(self, hand : list):
        """ 
        Get the rank of the 1 pair of a hand

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            rank (int): the rank of the 1 pair
        """
        # get 1 pair rank
        ranks = [card[1] for card in hand]
        arr, counts = np.unique(ranks, return_counts=True)
        rank = arr[np.where(counts == 2)][0]
        
        return 13 + rank

    def get_rank_full_house(self, hand : list):
        """ 
        Get the rank of the full house of a hand

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            rank (int): the rank of the full house
        """
        # get 3 of a kind rank
        ranks = [card[1] for card in hand]
        arr, counts = np.unique(ranks, return_counts=True)
        three_of_a_kind_rank = arr[np.where(counts == 3)][0]
        
        return 78 + three_of_a_kind_rank
    
    def get_rank_straight(self, hand : list):
        """ 
        Get the rank of the straight of a hand

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            rank (int): the rank of the straight
        """
        # get the highest card of the straight
        ranks = [card[1] for card in hand]
        ranks.sort()
        rank = ranks[-1]
        
        return 52 + rank
    
    def get_rank_flush(self, hand : list):
        """ 
        Get the rank of the flush of a hand

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            rank (int): the rank of the flush
        """
        # get the highest card of the flush
        suits = [card[0] for card in hand]
        suit = np.argmax(np.bincount(suits))
        ranks = [card[1] for card in hand if card[0] == suit]
        ranks.sort()
        rank = ranks[-1]
        
        return 65 + rank
    
    def get_rank_straight_flush(self, hand : list):
        """ 
        Get the rank of the straight flush of a hand

        Args:
            hand (list): a list of 2 tuples (cards)

        Returns:
            rank (int): the rank of the straight flush
        """
        # get the highest card of the straight flush
        suits = [card[0] for card in hand]
        suit = np.argmax(np.bincount(suits))
        ranks = [card[1] for card in hand if card[0] == suit]
        ranks.sort()
        rank = ranks[-1]
        
        return 104 + rank
    
    def get_rank_royal_flush(self, hand : list = None):
        """ 
        Get the rank of the royal flush of a hand

        Args:
            hand (list): a list of 2 tuples (cards)
            **just for consistency with other functions**

        Returns:
            rank (int): the rank of the royal flush
        """
        return 117
    
    def rank_hand_of_player(self, player : int, verbose : bool):
        """ 
        Rank the hand of a player.
        The hand of each player is a list of 2 tuples (cards).
        The tuple is of form (suit, rank).
        The rank of the hand is an integer from 0 to 12.
        The suit of the hand is an integer from 0 to 3.

        The rank of the hand is determined by the following rules:
        0 pairs --> rank is the highest rank of the hand
        1 pair --> rank is 13 + the rank of the pair
        2 pairs --> rank is 26 + the rank of the highest pair
        3 of a kind --> rank is 39 + the rank of the 3 of a kind
        Straight --> rank is 52 + the rank of the highest card
        Flush --> rank is 65 + the rank of the highest card of the flush
        Full House --> rank is 78 + the rank of the 3 of a kind
        4 of a kind --> rank is 91 + the rank of the 4 of a kind
        Straight Flush --> rank is 104 + the rank of the highest card of the straight flush
        Royal Flush --> rank is 117

        If any players have the same highest rank, we compare the second highest rank, and so on.

        Args:
            player (int): the player

        Returns:
            rank (int): the rank of the hand of the player
        """
        cards = self.player_hands[player] # 2 cards from player + 5 cards from table

        # loop through all possible hands, terminate when we find a hand. Start with the best hand, then go down
        hands = {self.is_royal_flush : self.get_rank_royal_flush,
                 self.is_straight_flush : self.get_rank_straight_flush,
                 self.is_4_of_a_kind : self.get_rank_of_4_of_a_kind,
                 self.is_full_house : self.get_rank_full_house,
                 self.is_flush : self.get_rank_flush,
                 self.is_straight : self.get_rank_straight,
                 self.is_3_of_a_kind : self.get_rank_of_3_of_a_kind,
                 self.is_2_pairs : self.get_rank_of_2_pairs,
                 self.is_1_pair : self.get_rank_of_1_pair,
                 self.is_high_card : self.get_rank_of_highest_card
        }
        hand_mapper = {
            self.get_rank_royal_flush : "Royal Flush",
            self.get_rank_straight_flush : "Straight Flush",
            self.get_rank_of_4_of_a_kind : "4 of a Kind",
            self.get_rank_full_house : "Full House",
            self.get_rank_flush : "Flush",
            self.get_rank_straight : "Straight",
            self.get_rank_of_3_of_a_kind : "3 of a Kind",
            self.get_rank_of_2_pairs : "2 Pairs",
            self.get_rank_of_1_pair : "1 Pair",
            self.get_rank_of_highest_card : "High Card"
        }   
        
        for fn_check, fn_rank in hands.items():
            is_hand, hand = fn_check(cards)
            if is_hand:
                map_hand = [self.mapper.card_string(card) for card in hand]
                if verbose:
                    print("-"*20)
                    print(f"Player {player+1} - {hand_mapper[fn_rank]}:")
                    hand_string = ", ".join(map_hand)
                    print(f"{hand_string}")
                    print("-"*20 + "\n")
                self.player_hands[player] = hand
                return fn_rank(hand)
        
        # if we get here, we have no hand, so the rank is the rank of the highest card
        return self.get_rank_of_highest_card(cards)
    
    def get_winner(self):
        """ 
        Find game winner.

        If any players have the same highest rank, we compare the second highest rank, and so on.
        """
        # first sort the players by their rank
        sorted_players = sorted(self.rank_player_hands, key=self.rank_player_hands.get, reverse=True)
        # check if any set of players have the same rank as the highest rank
        highest_rank = self.rank_player_hands[sorted_players[0]]
        
        players_with_highest_rank = []
        
        for player in sorted_players:
            if self.rank_player_hands[player] == highest_rank:
                players_with_highest_rank.append(player)
            else:
                break
        
        if len(players_with_highest_rank) == 1:
            return players_with_highest_rank
        
        # get the hands of the tied players
        player_hands = {player : self.player_hands[player] for player in players_with_highest_rank}
        # sort the hands of the tied players by rank of cards from hand + table cards
        sorted_players_hands = {player : sorted(player_hands[player], key=lambda card: card[1], reverse=True) for player in player_hands}
        # compare sorted ranks card by card until we find winner(s)
        players_removed = []
        
        for player1, hand1 in sorted_players_hands.items():
            for player2, hand2 in sorted_players_hands.items():
                if player1 == player2:
                    continue
                for i in range(5):
                    if hand1[i][1] > hand2[i][1]:
                        if player2 not in players_removed:
                            players_removed.append(player2)
                        break
                    if hand1[i][1] < hand2[i][1]:
                        if player1 not in players_removed:
                            players_removed.append(player1)
                        break
        
        for player in players_removed:
            players_with_highest_rank.remove(player)
        
        return players_with_highest_rank
                
