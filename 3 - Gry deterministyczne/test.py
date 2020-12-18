from copy import deepcopy
from random import randint

class Node:


    def __init__(self, board, starting_player): 
        self.board = board
        self.children = []
        self.starting_player = starting_player


    def get_heuristic(self):
        win = self.get_winner()
        if win == 1:
            return float("inf")
        if win == -1:
            return -float("inf")
        heuristic_table = [[3, 2, 3],
               [2, 4, 2],
               [3, 2, 3]]
        heur = 0
        for i in range(0,3):
            for k in range(0,3):
                heur += heuristic_table[i][k]*int(self.board[i][k])
        return heur


    def get_next_symbol(self):
        max_count = 0
        min_count = 0
        for row in self.board:
            max_count += row.count("1")
            min_count += row.count("-1")
        if max_count == min_count:
            return self.starting_player
        return str(-int(self.starting_player))

            
    def set_children(self):
        symbol = self.get_next_symbol()
        branches = []
        for row in range(3):
            for square in range(3):
                if self.board[row][square] == "0":
                    branches.append(deepcopy(self.board))
                    branches[-1][row][square] = symbol
        for branch in branches:
            self.children.append(Node(branch, self.starting_player))


    def get_winner(self):
        for i in range(0,3):
            sum = 0 
            for k in range(0,3):
                sum += int(self.board[i][k])
            if sum == 3:
                return 1
            if sum == -3:
                return -1

        for i in range(0,3):
            sum = 0 
            for k in range(0,3):
                sum += int(self.board[k][i])
            if sum == 3:
                return 1
            if sum == -3:
                return -1      
        sum = int(self.board[0][0]) + int(self.board[1][1]) + int(self.board[2][2])
        if sum == 3:
            return 1
        if sum == -3:
            return -1
        sum = int(self.board[0][2]) + int(self.board[1][1]) + int(self.board[2][0])
        if sum == 3:
            return 1
        if sum == -3:
            return -1
        return 0


    def is_terminal(self):
        if self.get_winner() != 0:
            return True
        for row in self.board:
            if "0" in row:
                return False
        return True


def print_board(tab, user_start):
    if user_start:
         dict = {"1":'x',
                "-1":'o', 
                "0": ' '}
    else:
        dict = {"1":'o',
                "-1":'x', 
                "0": ' '}

    print(f" 0    1    2")
    print(f"0 {dict[tab[0][0]]} | {dict[tab[0][1]]} | {dict[tab[0][2]]}")
    print(" ----------")
    print(f"1 {dict[tab[1][0]]} | {dict[tab[1][1]]} | {dict[tab[1][2]]}")
    print(" ----------")
    print(f"2 {dict[tab[2][0]]} | {dict[tab[2][1]]} | {dict[tab[2][2]]}")



def minimax(node, depth, max_player):
    node.set_children()
    if depth == 0 or node.is_terminal():
        return node.get_heuristic()
    if max_player:
        value = -float("inf")
        for child in node.children:
            value = max(value, minimax(child, depth - 1, False))
        return value
    else:
        value = float("inf")
        for child in node.children:
            value = min(value, minimax(child, depth - 1, True))
        return value


def game(user_start):
    start_board = [["0", "0", "0"],
               ["0", "0", "0"],
               ["0", "0", "0"]]
    if user_start:
        starting_player = "1"
    else: 
        starting_player = "-1"
    print_board(start_board, user_start)
    cur_node = Node(start_board, starting_player)
    user_turn = user_start
    while True:
        if cur_node.is_terminal():
            a = cur_node.get_winner()
            if a == 1:
                print("You won!")
                return
            if a == -1:
                print("You lost!")
                return
            if a == 0:
                print("Draw!")
                return
        if user_turn:
            board = cur_node.board
            print("Row: ")
            i = int(input())
            print("Place: ")
            k = int(input())
            board[i][k] = "1"
            cur_node = Node(board, starting_player)
            print_board(cur_node.board, user_start)
            user_turn  = False
        else:
            cur_node.set_children()
            min_p = float("inf")
            next_nodes = []

            for child in cur_node.children:
                val_minimax = minimax(child, 4, True)
                if val_minimax < min_p:
                    min_p = val_minimax
                    next_node = child
                elif val_minimax == min_p:
                    if randint(0,1) == 1:
                        min_p = val_minimax
                    next_node = child
            cur_node = next_node
            user_turn = True
            print_board(cur_node.board, user_start)


game(True) 