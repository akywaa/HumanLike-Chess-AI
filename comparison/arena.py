# note - script will not work unless you have the files `X_chess_data.npy`, `Y_chess_data.npy`, and `move_to_id.npy` in the project root (which are created by running the `prepare_data.py` script)

import chess
import chess.engine
import chess.pgn
import concurrent.futures
import datetime
import time
import sys
import os
import subprocess
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

GAMES_TO_PLAY = 10
PARALLEL_GAMES = 2
MAX_MOVES = 200

CUSTOM_ENGINE_CMD = [sys.executable, os.path.join(ROOT_DIR, "uci_engine.py")]
MAIA_ENGINE_CMD = [os.path.join(BASE_DIR, "lc0.exe"), f"--weights={os.path.join(BASE_DIR, 'maia-1800.pb.gz')}"]

def kill_zombie_processes():
    print("Cleaning up leftover engine processes...")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "lc0.exe", "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run('wmic process where "commandline like \'%uci_engine.py%\'" call terminate', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass
    time.sleep(2)

def play_one_game(game_idx):
    my_ai = None
    maia = None
    try:
        time.sleep(game_idx * 0.5)

        try:
            my_ai = chess.engine.SimpleEngine.popen_uci(CUSTOM_ENGINE_CMD)
        except Exception as e:
            return {"error": f"[MyResNet Error] {e}\n{traceback.format_exc()}", "id": game_idx + 1}

        try:
            maia = chess.engine.SimpleEngine.popen_uci(MAIA_ENGINE_CMD)
        except Exception as e:
            return {"error": f"[Maia Error] {e}\n{traceback.format_exc()}", "id": game_idx + 1}

        if game_idx % 2 == 0:
            white_name = "MyResNet-1900"
            black_name = "Maia-1800"
            white_engine = my_ai
            black_engine = maia
        else:
            white_name = "Maia-1800"
            black_name = "MyResNet-1900"
            white_engine = maia
            black_engine = my_ai

        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = f"AI Battle: MyResNet vs Maia"
        game.headers["Round"] = str(game_idx + 1)
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")

        node = game

        while not board.is_game_over(claim_draw=True) and board.fullmove_number <= MAX_MOVES:
            active_engine = white_engine if board.turn == chess.WHITE else black_engine
            active_name = white_name if board.turn == chess.WHITE else black_name

            if active_name == "Maia-1800":
                limit = chess.engine.Limit(nodes=1)
            else:
                limit = chess.engine.Limit(time=0.1)

            result = active_engine.play(board, limit)
            if result.move is None:
                break

            board.push(result.move)
            node = node.add_variation(result.move)

        if board.fullmove_number > MAX_MOVES:
            final_result = "1/2-1/2"
        else:
            final_result = board.result(claim_draw=True)

        game.headers["Result"] = final_result

        return {
            "id": game_idx + 1,
            "white": white_name,
            "black": black_name,
            "result": final_result,
            "pgn": str(game)
        }
    except Exception as e:
        return {"error": f"[Runtime Error] {e}", "id": game_idx + 1}
    finally:
        if my_ai is not None:
            try:
                my_ai.quit()
            except:
                pass
        if maia is not None:
            try:
                maia.quit()
            except:
                pass

if __name__ == "__main__":
    kill_zombie_processes()

    print(f"Match start: MyResNet vs Maia")
    print(f"Games: {GAMES_TO_PLAY}. Parallel: {PARALLEL_GAMES}\n")

    results_list = []
    my_wins = 0
    maia_wins = 0
    draws = 0

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_GAMES) as executor:
        futures = [executor.submit(play_one_game, i) for i in range(GAMES_TO_PLAY)]

        for future in concurrent.futures.as_completed(futures):
            res = future.result()

            if "error" in res:
                print(f"[Game {res['id']}] Error:\n{res['error']}")
                continue

            r = res["result"]
            white = res["white"]

            if r == "1-0":
                if white == "MyResNet-1900": my_wins += 1
                else: maia_wins += 1
            elif r == "0-1":
                if white == "MyResNet-1900": maia_wins += 1
                else: my_wins += 1
            else:
                draws += 1

            results_list.append(res)
            print(f"Game {res['id']} finished. Score: MyResNet [{my_wins} - {maia_wins}] Maia (Draws: {draws})")

    print(f"\nMatch finished in {round(time.time() - start_time, 1)} seconds.")

    results_path = os.path.join(BASE_DIR, "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== Final match result ===\n")
        f.write(f"MyResNet-1900 wins: {my_wins}\n")
        f.write(f"Maia-1800 wins: {maia_wins}\n")
        f.write(f"Draws: {draws}\n")
        f.write("================================\n\n")

        results_list.sort(key=lambda x: x.get("id", 0))

        for res in results_list:
            if "pgn" in res:
                f.write(res["pgn"])
                f.write("\n\n")

    print("All stats and PGN games saved to 'results.txt'.")