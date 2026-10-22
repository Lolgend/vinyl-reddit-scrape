# Time-Series Velocity for Hype Scoring

Newly announced releases initially have zero comments and a single upvote, making a one-time static metric check ineffective for detecting real community hype. We decided to capture periodic timestamped snapshots of engagement metrics and compute velocity ($\Delta \text{score}/\Delta t$, $\Delta \text{comments}/\Delta t$) with time-decay weighting. This surfaces breaking drops and surging re-presses based on momentum rather than cumulative age bias.
