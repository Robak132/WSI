from copy import deepcopy
import random


class TicTacToe:
    def __init__(self, maxlevel, aiturn=None, maximizing=True, board=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]):
        self.maxlevel = maxlevel
        self.board = board
        self.maximizing = maximizing
        if aiturn is None:
            self.aiturn = random.choice([True, False])
        else:
            self.aiturn = aiturn
        self.value = 0

    def create_children(self):
        self.children = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    temp_board = deepcopy(self.board)
                    if self.maximizing:
                        temp_board[i][j] = 1
                    else:
                        temp_board[i][j] = -1
                    self.children.append(TicTacToe(self.maxlevel, self.aiturn, not self.maximizing, temp_board))

    def play(self):
        if self.aiturn:
            print("Zaczyna AI")
        else:
            print("Zaczyna gracz")
            self.show()

        while not self.check_terminal():
            if self.aiturn:
                self.do()
                self.show()
            else:
                move = self.getinput()
                self.do(move)
                self.show()
        if self.get_winner() == 1:
            print("Player X won!")
        elif self.get_winner() == -1:
            print("Player O won!")
        else:
            print("Draw!")

    def getinput(self):
        valid_move = False
        while(not valid_move):
            try:
                temp_inp = input("Twój ruch (x, y):\n")
                temp_inp = temp_inp.split(",")
                temp_inp = tuple([int(xy) for xy in temp_inp])
            except(Exception):
                print("Zły format danych")
                continue

            if not self.validate_move(temp_inp):
                print("Niepoprawny ruch")
            else:
                valid_move = self.validate_move(temp_inp)
        return temp_inp

    def validate_move(self, move):
        if move not in self.get_possible_moves():
            return False
        else:
            return True

    def do(self, xy=None, board=None):
        # AI plays
        if xy is None and board is None:
            self.value = minimax(self, self.maxlevel, self.maximizing)
            new_board = self.next_move()
            self.board = new_board
        # Human player plays
        else:
            self.board[xy[1]][xy[0]] = -1

        self.aiturn = not self.aiturn

    def next_move(self):
        random.shuffle(self.children)
        return sorted(self.children, reverse=self.maximizing, key=lambda x: x.value)[0].board

    def get_possible_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[j][i] == 0:
                    moves.append((i, j))
        return moves

    def get_winner(self):
        if abs(self.board[0][0] + self.board[1][1] + self.board[2][2]) == 3:
            return self.board[0][0]
        if abs(self.board[2][0] + self.board[1][1] + self.board[0][2]) == 3:
            return self.board[2][0]

        for i in range(3):
            if abs(self.board[0][i] + self.board[1][i] + self.board[2][i]) == 3:
                return self.board[0][i]
            if abs(self.board[i][0] + self.board[i][1] + self.board[i][2]) == 3:
                return self.board[i][0]

        return 0

    def check_terminal(self):
        if self.get_winner() != 0:
            return True
        if self.get_possible_moves() == []:
            return True
        return False

    def show(self):
        print(f"\\ 0 1 2")
        for i in range(len(self.board)):
            print(i, end=" ")
            for n in self.board[i]:
                if n == 1:
                    print("X", end=" ")
                elif n == -1:
                    print("O", end=" ")
                else:
                    print("-", end=" ")
            print()
        print()

    def get_approx_value(self):
        if self.check_terminal():
            if self.get_winner() != 0:
                return self.get_winner() * float("inf")
            else:
                return 0
        else:
            approx = self.board[1][1] * 4
            approx += (self.board[0][0] + self.board[0][2] + self.board[2][2] + self.board[2][0]) * 3
            approx += (self.board[0][1] + self.board[1][2] + self.board[2][1] + self.board[1][0]) * 2
            return approx

    def __repr__(self):
        return f"{str(self.board)};{self.value}"


def minimax(node, level, maximizingPlayer):
    node.create_children()
    if node.check_terminal() or level == 0:
        node.value = node.get_approx_value()
        return node.get_approx_value()
    elif maximizingPlayer:
        node.value = -float("inf")
        for child in node.children:
            val = minimax(child, level - 1, False)
            node.value = max(node.value, val)
        return node.value
    else:
        node.value = float("inf")
        for child in node.children:
            val = minimax(child, level - 1, True)
            node.value = min(node.value, val)
        return node.value


if __name__ == "__main__":
    t = TicTacToe(5)
    t.play()
