# N-Former F1 Car: The Driver's Manual

Welcome to the cockpit of your very own N-Former F1 car! This is not just any model; it's a powerful AI machine, and you are the driver. This manual will help you understand the "gears" and "dials" at your disposal to control this advanced machine and get the best performance out of it.

## The Main Gears:

### `temperature`: The Creativity Dial

*   **What it is:** This is your main dial for controlling the creativity and randomness of the engine's output.
*   **Influence:**
    *   **Low temperature (e.g., 0.8):** Like driving in a straight line. The car is stable, predictable, and coherent. Perfect for when you need to stay on track.
    *   **High temperature (e.g., 1.4):** This is like hitting the nitrous button. The car becomes wildly creative and unpredictable, capable of amazing feats of imagination, but also at risk of spinning out of control.
*   **Trade-off:** Coherence vs. Creativity.

### `top_k`: The Vocabulary Filter

*   **What it is:** This is your vocabulary filter. It limits the engine's word choices to the `k` most likely options at each step.
*   **Influence:**
    *   **Low `top_k` (e.g., 10):** Like driving with a limited set of gears. The car is more predictable and less likely to do something unexpected, even at high speeds (high temperature).
    *   **High `top_k` (e.g., 200):** This is like having all the gears at your disposal. The car has more freedom to choose its path.
*   **Trade-off:** Predictability vs. Diversity.

## The Fine-Tuning Gears:

### `seed`: The Reproducibility Switch

*   **What it is:** This is your "instant replay" button. It controls the random number generator.
*   **Influence:** Using the same `seed` with the same settings will give you the exact same performance every time. This is great for practicing your driving and reproducing those perfect laps.

### Mathematical Enhancement Parameters: The "Under the Hood" Gears

These are the advanced settings that our race engineers have built into the car. Their effects are subtle, but they can give you that extra edge on the track.

*   **`pi_amp` & `pi_freq` (The Rhythm Section):** These control the "Pi-Delta Oscillation," which adds a subtle rhythmic hum to the engine. This can influence the "voice" and "flow" of the car's performance.
*   **`infinity_threshold` (The Confidence Gate):** This controls the "Infinity Gating" mechanism. A higher threshold makes the car more "confident" in its maneuvers, sticking to the lines it knows best.

---

Now that you have the driver's manual, you are ready to take this F1 car for a spin. Remember, the best way to learn is to practice. Experiment with these gears and see what this amazing machine can do!
