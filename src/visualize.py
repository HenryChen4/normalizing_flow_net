import numpy as np

def visualize(solution, link_lengths, objective, ax):
    """Plots an arm with the given angles and link lengths on ax.
    
    Args:
        solution (np.ndarray): A (dim,) array with the joint angles of the arm.
        link_lengths (np.ndarray): The length of each link the arm.
        objective (float): The objective of this solution.
        ax (plt.Axes): A matplotlib axis on which to display the arm.
    """
    lim = 1.05 * np.sum(link_lengths)  # Add a bit of a border.
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_title(f"Objective: {objective}")

    # Plot each link / joint.
    pos = np.array([0, 0])  # Starting position of the next joint.
    cum_thetas = np.cumsum(solution)
    for link_length, cum_theta in zip(link_lengths, cum_thetas):
        # Calculate the end of this link.
        next_pos = pos + link_length * np.array(
            [np.cos(cum_theta), np.sin(cum_theta)])
        ax.plot([pos[0], next_pos[0]], [pos[1], next_pos[1]], "-ko", ms=3)
        pos = next_pos

    # Add points for the start and end positions.
    ax.plot(0, 0, "ro", ms=6)
    final_label = f"Final: ({pos[0]:.2f}, {pos[1]:.2f})"
    ax.plot(pos[0], pos[1], "go", ms=6, label=final_label)
    ax.legend()