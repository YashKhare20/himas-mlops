# himas-model-pipeline: A Flower / TensorFlow app

Federated learning application for ICU mortality prediction with server-side differential privacy.

## Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

> **Tip:** Your `pyproject.toml` file can define more than just the dependencies of your Flower app. You can also use it to specify hyperparameters for your runs and control which Flower Runtime is used. By default, it uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

## Run with the Simulation Engine

In the `himas-model-pipeline` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Differential Privacy

This application includes server-side differential privacy to protect patient data during federated learning.

### Configuration

Edit `pyproject.toml` to enable/disable or adjust privacy settings:

```toml
[tool.flwr.app.config]
enable-differential-privacy = true  # Toggle DP on/off
dp-noise-multiplier = 1.0          # Privacy level (higher = more privacy)
dp-clipping-norm = 1.0             # Gradient clipping threshold
dp-num-sampled-clients = 3         # Number of clients per round
random-seed = 42                   # For reproducibility
```

**Quick Guide:**
- Set `enable-differential-privacy = false` for baseline accuracy
- Set `enable-differential-privacy = true` for privacy-preserving training
- Adjust `dp-noise-multiplier`: `0.5` (less privacy) to `2.0` (more privacy)

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
