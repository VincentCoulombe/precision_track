# Weight & Biases guide

## What W&B does (experiment tracking)

- **Log everything, quickly**
  - Metrics, losses, hyperparameters, system info, gradients, media (images/audio/video), and artifacts.
  - Learn more: [Experiment tracking overview](https://wandb.ai/site/experiment-tracking), [Tracking guide](https://docs.wandb.ai/guides/track)
- **Compare runs at scale**
  - Interactive dashboards, tables, and reports to slice, filter, and compare experiments.
- **Comes integrated with PrecisionTrack**
  - Works out of the box.

______________________________________________________________________

## Step 1: Register an account

In order to leverage a Weight & Biases tracking service, you must first register a free account to their website. You can do so by doing the following:

1. **Sign up:** go to [wandb.ai](https://wandb.ai/) and click **Sign Up** (GitHub/Google/Microsoft/email supported).
1. **Create your API key:** after sign‑up, your key is available in **User Settings → API Keys**.
1. **Log in on your machine:**
   - In a terminal:
   ```bash
        wandb login <YOUR_API_KEY>
   ```

## Step 2: Configure PrecisionTrack

Once you have registered to Weight & Biases, you need a way for PrecisionTrack to connect to your account. To do so, you have to perform the following:

1. **Configure ./keys.py** open up ./keys.py and fill the two mandatory fields:
   - entity: Your Weight & Biases username.
   - project: The name you want your runs to take within Weight & Biases.

## Step 3: Enjoy

You will now be able to track your training runs, compare them directly and deploy the most peformant checkpoints.

______________________________________________________________________

<details>
<summary>Official documentation (handy links)</summary>

- [Docs home](https://docs.wandb.ai/)
- [Experiment tracking](https://wandb.ai/site/experiment-tracking)
- [Tracking guide](https://docs.wandb.ai/guides/track)
- [Quickstart](https://docs.wandb.ai/quickstart)
- [Python `wandb.login`](https://docs.wandb.ai/ref/python/python_api_walkthrough/)

</details>
