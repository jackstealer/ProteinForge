import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import sys


class TicTacToeGame:

    def __init__(self):
        self.board = ['b'] * 9
        self.current_player = 'x'
        self.model = None
        self.label_encoders = {}
        self.game_over = False
        self.winner = None

        self.position_names = [
            'top-left-square', 'top-middle-square', 'top-right-square',
            'middle-left-square', 'middle-middle-square', 'middle-right-square',
            'bottom-left-square', 'bottom-middle-square', 'bottom-right-square'
        ]

    def train_ai(self, filepath='tic_tac_toc.csv'):
        print("\n🤖 Training AI...")

        df = pd.read_csv(filepath)

        df.columns = [col.strip() for col in df.columns]
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace("b'", "").str.replace("'", "")

        for col in df.columns[:-1]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        le_target = LabelEncoder()
        df['Class'] = le_target.fit_transform(df['Class'])
        self.label_encoders['Class'] = le_target

        X = df.drop('Class', axis=1)
        y = df['Class']

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        print("✅ AI trained successfully!")
        print(f"   Model accuracy: {self.model.score(X, y):.2%}")
        print("🧠 AI Strategy: Minimax Algorithm with Alpha-Beta Pruning")
        print("💪 Difficulty: EXTREME - Plays perfectly!")
        print("⚠️  Warning: Nearly impossible to win!")

    def display_board(self):
        print("\n" + "="*40)
        print("         TIC-TAC-TOE BOARD")
        print("="*40)

        print("\n  Positions:              Current Board:")
        for i in range(3):
            pos_line = f"   {i*3+1} | {i*3+2} | {i*3+3}              "
            board_line = f" {self.board[i*3]} | {self.board[i*3+1]} | {self.board[i*3+2]}"
            print(pos_line + board_line)
            if i < 2:
                print("  -----------            -----------")
        print()

    def check_winner(self):
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]

        for combo in winning_combos:
            if (self.board[combo[0]] == self.board[combo[1]] ==
                self.board[combo[2]] != 'b'):
                self.winner = self.board[combo[0]]
                return True

        if 'b' not in self.board:
            self.winner = 'draw'
            return True

        return False

    def make_move(self, position):
        if position < 0 or position > 8:
            return False
        if self.board[position] != 'b':
            return False

        self.board[position] = self.current_player
        return True

    def get_board_state(self):
        state = {}
        for i, pos_name in enumerate(self.position_names):
            state[pos_name] = self.board[i]
        return state

    def minimax(self, board, depth, is_maximizing, alpha, beta):
        if self._check_win(board, 'o'):
            return 10 - depth
        if self._check_win(board, 'x'):
            return depth - 10
        if 'b' not in board:
            return 0

        available_positions = [i for i, val in enumerate(board) if val == 'b']

        if is_maximizing:
            max_score = -float('inf')
            for pos in available_positions:
                temp_board = board.copy()
                temp_board[pos] = 'o'
                score = self.minimax(temp_board, depth + 1, False, alpha, beta)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = float('inf')
            for pos in available_positions:
                temp_board = board.copy()
                temp_board[pos] = 'x'
                score = self.minimax(temp_board, depth + 1, True, alpha, beta)
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_score

    def ai_move(self):
        print("\n🤖 AI is analyzing all possible outcomes...")

        available_positions = [i for i, val in enumerate(self.board) if val == 'b']

        if not available_positions:
            return None

        if self.board.count('b') == 9:
            return 4

        if self.board.count('b') == 8:
            if self.board[4] == 'x':
                return 0
            return 4

        best_score = -float('inf')
        best_position = None

        for pos in available_positions:
            temp_board = self.board.copy()
            temp_board[pos] = 'o'

            score = self.minimax(temp_board, 0, False, -float('inf'), float('inf'))

            if score > best_score:
                best_score = score
                best_position = pos

        return best_position if best_position is not None else available_positions[0]

    def _check_win(self, board, player):
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for combo in winning_combos:
            if (board[combo[0]] == board[combo[1]] ==
                board[combo[2]] == player):
                return True
        return False

    def play_game(self):
        print("\n" + "="*60)
        print("      🔥 EXTREME DIFFICULTY TIC-TAC-TOE 🔥")
        print("="*60)
        print("\n  ⚠️  AI uses PERFECT strategy (Minimax Algorithm)")
        print("  🎯 You are X, AI is O")
        print("  💡 TIP: A draw is a great result against this AI!")
        print("\n" + "="*60)

        print("\nWho should go first?")
        print("  1. You (X)")
        print("  2. AI (O)")

        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("❌ Invalid choice! Please enter 1 or 2.")

        if choice == '2':
            self.current_player = 'o'

        while not self.game_over:
            self.display_board()

            if self.current_player == 'x':
                print("👤 Your turn (X)")
                while True:
                    try:
                        move = input("Enter position (1-9) or 'q' to quit: ").strip().lower()
                        if move == 'q':
                            print("\n👋 Thanks for playing!")
                            return

                        position = int(move) - 1
                        if self.make_move(position):
                            break
                        else:
                            print("❌ Invalid move! Position already taken or out of range.")
                    except ValueError:
                        print("❌ Please enter a number between 1-9.")
            else:
                position = self.ai_move()
                if position is not None:
                    self.make_move(position)
                    print(f"🤖 AI plays position {position + 1}")

            if self.check_winner():
                self.game_over = True
                self.display_board()

                print("\n" + "="*60)
                if self.winner == 'x':
                    print("  🎉 CONGRATULATIONS! YOU WIN! 🎉")
                elif self.winner == 'o':
                    print("  🤖 AI WINS! Better luck next time!")
                else:
                    print("  🤝 IT'S A DRAW!")
                print("="*60)
            else:
                self.current_player = 'o' if self.current_player == 'x' else 'x'

    def play_again(self):
        while True:
            choice = input("\n🎮 Play again? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            print("❌ Please enter 'y' or 'n'.")


def main():
    print("\n" + "="*60)
    print("  🎮 WELCOME TO INTELLIGENT TIC-TAC-TOE! 🎮")
    print("="*60)
    print("\n  ⚠️  EXTREME DIFFICULTY MODE ⚠️")
    print("  The AI uses MINIMAX algorithm - Nearly IMPOSSIBLE to beat!")
    print("  It calculates ALL possible game outcomes.")
    print("  Best you can hope for is a DRAW!")
    print("\n" + "="*60)

    if not os.path.exists('tic_tac_toc.csv'):
        print("\n❌ Error: tic_tac_toc.csv not found!")
        print("   Please make sure the dataset is in the same folder.")
        return

    game = TicTacToeGame()

    game.train_ai()

    while True:
        game.board = ['b'] * 9
        game.current_player = 'x'
        game.game_over = False
        game.winner = None

        game.play_game()

        if not game.play_again():
            print("\n" + "="*60)
            print("  👋 Thank you for playing!")
            print("  🎮 Come back soon!")
            print("="*60 + "\n")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted. Thanks for playing!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("   Please check that tic_tac_toc.csv is in the correct location.")
