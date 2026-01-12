module.exports = {
  run: [
    // Install PyTorch with CUDA support first
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv_python: "3.12",
          venv: "env",
          xformers: true,
          flashattn: true
        }
      }
    },
    // Install Liquid Audio dependencies from requirements.txt (PyTorch already installed from torch.js)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt --no-deps",
          "uv pip install gradio transformers accelerate librosa sentencepiece"
        ],
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch Liquid Audio. Models will be downloaded automatically from Hugging Face on first use."
      }
    }
  ]
}
