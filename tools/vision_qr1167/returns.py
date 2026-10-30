"""Returns for finite driving attempts; never join different real episodes."""


def driving_return(rows, episode_ids, index, horizon=5, gamma=.997):
    observation, action, _, _, _, _ = rows[index]
    episode = episode_ids[index]
    reward = 0.
    discount = 1.
    end = False
    steps = 0
    for offset in range(horizon if episode is not None else 1):
        position = index+offset
        if position >= len(rows) or (offset and episode_ids[position] != episode):
            break
        _, _, value, following, goal, reset = rows[position]
        reward += discount*value
        steps += 1
        # Stall, corridor exit and finite-attempt timeout end this MDP attempt.
        # Their original flags stay in the replay; this is not a NativeGoal.
        end = bool(goal or reset)
        if end:
            discount = 0.
            break
        discount *= gamma
    return observation, action, reward, following, end, discount, steps
