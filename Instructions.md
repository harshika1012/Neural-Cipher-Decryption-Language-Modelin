# Instructions

Ensure you have `uv` installed. Then run `uv sync` to create the environment.

code runs from the `main.py` file using the task subcommands:

```bash
uv run main.py <task> [--config CONFIG_PATH] [--mode {train,evaluate,both}]
```

Available tasks:

| Subcommand     | Description                              | Default Config              |
|----------------|------------------------------------------|-----------------------------|
| `task1_rnn`    | Train/evaluate RNN for decryption        | `config/task1/rnn.yaml`     |
| `task1_lstm`   | Train/evaluate LSTM for decryption       | `config/task1/lstm.yaml`    |
| `task2_bilstm` | Train/evaluate Bi-LSTM for MLM           | `config/task2/bilstm.yaml`  |
| `task2_ssm`    | Train/evaluate SSM for NWP               | `config/task2/ssm.yaml`     |
| `task3_bilstm` | Error correction pipeline using bilstm   | `config/task3/bilstm.yaml`  |
| `task3_ssm`    | Error correction pipeline using ssm      | `config/task3/ssm.yaml`     |

Available modes:

| Mode       | Description                               |
|------------|-------------------------------------------|
| `train`    | Train the model                           |
| `evaluate` | Run inference on trained model (default)  |
| `both`     | Train and then evaluate                   |

Examples:

```bash
uv run main.py task1_rnn
uv run main.py task1_rnn --mode train
uv run main.py task1_rnn --mode evaluate
uv run main.py task1_lstm --mode both
uv run main.py task2_ssm --mode evaluate
uv run main.py task3_bilstm
```

