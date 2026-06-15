
import os
import numpy as np
import matplotlib.pyplot as plt

def save_models_and_generate_plots(jerry_agent, tom_agent, metrics_tracker):
    os.makedirs('/content/drive/MyDrive/rl-gridworld/models', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/rl-gridworld/assets/graphs', exist_ok=True)

    jerry_q_data = {str(k): np.array(v) for k, v in jerry_agent.q_table.items()}
    tom_q_data   = {str(k): np.array(v) for k, v in tom_agent.q_table.items()}

    np.save('/content/drive/MyDrive/rl-gridworld/models/qtable_jerry.npy', jerry_q_data)
    np.save('/content/drive/MyDrive/rl-gridworld/models/qtable_tom.npy', tom_q_data)
    print("[SUCCESS] qtable_jerry.npy and qtable_tom.npy saved!")

    total_episodes = len(metrics_tracker.winners)
    window_size = 500

    episodes_axis = []
    jerry_win_rates = []
    tom_win_rates = []
    draw_rates = []
    avg_rewards_jerry = []

    for i in range(window_size, total_episodes + 1, window_size):
        window_winners = metrics_tracker.winners[i - window_size : i]
        window_jerry_rewards = metrics_tracker.episode_rewards_jerry[i - window_size : i]

        episodes_axis.append(i)
        jerry_win_rates.append((window_winners.count('jerry') / window_size) * 100)
        tom_win_rates.append((window_winners.count('tom') / window_size) * 100)
        draw_rates.append((window_winners.count('none') / window_size) * 100)
        avg_rewards_jerry.append(sum(window_winners) / window_size if isinstance(window_winners[0], (int, float)) else sum(window_jerry_rewards)/window_size)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_axis, jerry_win_rates, label="Jerry Win Rate (%)", color="orange", linewidth=2)
    plt.plot(episodes_axis, tom_win_rates, label="Tom Win Rate (%)", color="blue", linewidth=2)
    plt.plot(episodes_axis, draw_rates, label="Draw Rate (%)", color="gray", linestyle=":", alpha=0.7)

    plt.title("Agent Win Rates Progression", fontsize=12, fontweight='bold')
    plt.xlabel("Episodes", fontsize=10)
    plt.ylabel("Percentage (%)", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)

    plot1_path = '/content/drive/MyDrive/rl-gridworld/assets/graphs/win_rate_progression.png'
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[SUCCESS] save win-rate graph as: {plot1_path}")

    avg_lengths = []
    for i in range(window_size, total_episodes + 1, window_size):
        avg_lengths.append(sum(metrics_tracker.episode_lengths[i - window_size : i]) / window_size)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_axis, avg_lengths, color="purple", linewidth=2, label="Avg Steps per Episode")
    plt.title("Learning Convergence (Average Episode Length)", fontsize=12, fontweight='bold')
    plt.xlabel("Episodes", fontsize=10)
    plt.ylabel("Steps", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)

    plot2_path = '/content/drive/MyDrive/rl-gridworld/assets/graphs/episode_length.png'
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[SUCCESS] save game duration as: {plot2_path}")
