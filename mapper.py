
class Mapper:
    """ 
    Mapper class for mapping the integer representation of a hand
    to a string representation of the hand for readability.
    
    Cards are 0-12: 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace
    Suits are 0-3: Clubs, Diamonds, Hearts, Spades
    """
    def __init__(self):
        self.suits = ["♣", "♦", "♥", "♠"]
        self.ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", 
                      "J", "Q", "K", "A"]
    
    def card_string(self, card : tuple):
        """ 
        Get the string representation of a card
        
        Args:
            card (tuple): a card as tuple (suit, rank)
        
        Returns:
            card_string (str): the string representation of the card
        """
        return self.ranks[card[1]] + self.suits[card[0]]