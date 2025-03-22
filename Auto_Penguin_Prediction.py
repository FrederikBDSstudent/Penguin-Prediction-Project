name: Daily Penguin Prediction

on:
  schedule:
    - cron: '30 5 * * *'  # Runs at 05:30 UTC = 07:30 CEST
  workflow_dispatch:      # Allows manual runs from GitHub

jobs:
  predict:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run prediction script
        run: python Auto_Penguin_Prediction.py

      - name: Commit and push prediction result
        uses: EndBug/add-and-commit@v9
        with:
          message: "Daily penguin prediction at 07:30 (DK time)"
          add: 'data/prediction_result.json'
          default_author: github_actions
